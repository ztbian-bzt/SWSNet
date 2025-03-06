import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Optional
from itertools import filterfalse
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


class LovaszLoss(_Loss):
    def __init__(
        self,
        class_seen: Optional[int] = None,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        """Lovasz loss for segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()

        self.ignore_index = ignore_index
        self.per_image = per_image
        self.class_seen = class_seen
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        y_pred = y_pred.softmax(dim=1)
        loss = _lovasz_softmax(
                y_pred,
                y_true,
                class_seen=self.class_seen,
                per_image=self.per_image,
                ignore=self.ignore_index,
            )
        return loss * self.loss_weight


def _lovasz_softmax(
    probas, labels, classes="present", class_seen=None, per_image=False, ignore=None
):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probas(probas, labels, ignore),
            classes=classes,
            class_seen=class_seen
        )
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present", class_seen=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    # for c in class_to_sum:

    for c in labels.unique():
        if class_seen is None:
            fg = (labels == c).type_as(probas)  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        else:
            if c in class_seen:
                fg = (labels == c).type_as(probas)  # foreground for class c
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, c]
                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                perm = perm.data
                fg_sorted = fg[perm]
                losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    # if probas.dim() == 3:
    #     # assumes output of a sigmoid layer
    #     B, H, W = probas.size()
    #     probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nan-mean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="none"):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i)
                          for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i)
                          for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()

class InstMatcher(nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.beta = 0.2
        self.mask_score = dice_score

    def forward(self, outputs, targets):
        with torch.no_grad():
            B, N, _ = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()

            tgt_ids = []
            tgt_masks = []
            for batch_idx in range(targets.shape[0]):
                target = targets[batch_idx]
                for lbl in sorted(target.unique()):
                    if lbl == 16:  # 16 is background
                        continue
                    tgt_ids.append(lbl.unsqueeze(0))
                    tgt_mask = torch.where(target == lbl, 1, 0)
                    tgt_masks.append(tgt_mask.unsqueeze(0))
            tgt_ids = torch.cat(tgt_ids)
            tgt_masks = torch.cat(tgt_masks).to(pred_masks) #.to(xxx) 类型转换

            pred_masks = pred_masks.view(B * N, -1)

            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                # cv 0527
                matching_prob = pred_logits.reshape(B * N, -1)[:, tgt_ids]

                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v.unique())-1 for v in targets] # 不包括背景
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices, tgt_masks

class Cal_Loss(nn.Module):
    def __init__(self):
        super(Cal_Loss, self).__init__()

    def forward(self, outputs, targets):
        # matcher
        matcher = InstMatcher()
        indices, target_masks = matcher(outputs, targets)
        num_instances = torch.as_tensor([sum(len(v.unique())-1 for v in targets)], dtype=torch.float)

        # loss_labels
        src_logits = outputs['pred_logits']
        idx = get_src_permutation_idx(indices)
        targets_labels = [target.unique() for target in targets]
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets_labels, indices)])
        target_classes = torch.full(src_logits.shape[:2], 16, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)

        pos_inds = torch.nonzero(target_classes != 16, as_tuple=True)[0]  # 16 is the class number without background
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1

        loss_labels = sigmoid_focal_loss(src_logits, labels, alpha=0.25, gamma=2.0, reduction="sum") / num_instances.cuda()

        # loss_objectness
        src_idx = get_src_permutation_idx(indices)
        tgt_idx = get_tgt_permutation_idx(indices)
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]

        src_masks = src_masks[src_idx]
        src_masks = src_masks.flatten(1)
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        num_masks = [len(v.unique()) - 1 for v in targets]
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]
        target_masks = target_masks[mix_tgt_idx].flatten(1)

        ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)


        loss_objectness = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
        # loss_masks
        loss_dice = dice_loss(src_masks, target_masks) / num_instances.cuda()
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        # loss_masks = 1.0 * loss_dice + 2.0 * loss_mask
        loss_masks = 2.0 * loss_dice + 5.0 * loss_mask

        # loss_all = loss_labels + loss_objectness + loss_masks
        # lambda
        loss_all = 2.0 * loss_labels + 1.0 * loss_masks + 1.0 * loss_objectness
        return loss_all