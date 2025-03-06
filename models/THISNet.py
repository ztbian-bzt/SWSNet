import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchmetrics as tm
import lightning as L
from models.loss import Cal_Loss


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, k=10):
    batch_size, num_dims, num_points  = coor.shape
    coor = coor.view(batch_size, -1, num_points)

    idx = knn(coor, k=k)
    index = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()

    coor = coor.transpose(2,1).contiguous()
    nor = nor.transpose(2,1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index


class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature


class THISNet(nn.Module):
    def __init__(self, k=32, in_channels=24, output_channels=8):
        super(THISNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv2_c = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv3_c = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv4_c = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=128, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=12)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv2_n = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv3_n = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))



        self.conv4_n = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.FTM_n1 = STNkd(k=12)

        '''feature-wise attention'''

        self.fa = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2))

        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        # self.dp1 = nn.Dropout(p=0.6)
        # self.dp2 = nn.Dropout(p=0.6)
        # self.dp3 = nn.Dropout(p=0.6)


        # instance attention map
        self.fuseconv = nn.Conv1d(1024, 128, kernel_size=1, bias=False)
        self.iam_conv = nn.Conv1d(128, 120, kernel_size=1, bias=False)
        self.fc = nn.Conv1d(512, 128, kernel_size=1, bias=False)
        # outputs
        self.cls_score1 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.cls_score2 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        self.mask_kernel = nn.Linear(128, 128)
        self.objectness = nn.Linear(128, 1)

        self.projection = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.output_iam = True

    def forward(self, x):
        coord = x[:, :12, :]
        nor = x[:, 12:, :]
        batch_size, num = coord.shape[0], coord.shape[2]
        # transform
        trans_c = self.FTM_c1(coord)
        coor = coord.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]

        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)

        avgSum_coor = coor.sum(1)/512
        avgSum_nor = nor.sum(1)/512
        avgSum = avgSum_coor+avgSum_nor
        weight_coor = (avgSum_coor / avgSum).reshape(batch_size,1,num)
        weight_nor = (avgSum_nor / avgSum).reshape(batch_size,1,num)
        x = torch.cat((coor*weight_coor, nor*weight_nor), dim=1)

        x = self.fa(x)
        # x = torch.cat((x, coord), dim=1)  # location aware

        # instance branch
        x_ = self.fuseconv(x)
        iam = self.iam_conv(x_)
        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]

        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        inst_features = torch.bmm(iam_prob, x_.permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, 4, N // 4, -1).transpose(1, 2).reshape(B, N // 4, -1)

        inst_features = F.relu_(self.fc(inst_features.transpose(2,1)))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score1(inst_features)
        pred_logits = self.cls_score2(pred_logits)
        pred_logits = pred_logits.transpose(2,1)
        pred_kernel = self.mask_kernel(inst_features.transpose(2,1))
        pred_scores = self.objectness(inst_features.transpose(2,1))

        # mask branch
        mask_features = self.projection(x_)
        pred_masks = torch.bmm(pred_kernel, mask_features)

        output = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }
        if self.output_iam:
            output['pred_iam'] = iam

        # auxiliary semantic branch
        x = self.pred1(x)
        # self.dp1(x)
        x = self.pred2(x)
        # self.dp2(x)
        x = self.pred3(x)
        # self.dp3(x)
        score = self.pred4(x)
        # score = F.log_softmax(score, dim=1)
        # score = score.permute(0, 2, 1)

        return output, score


class LitTHISNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = THISNet(k=32, in_channels=24, output_channels=17)
        self.loss = Cal_Loss()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_dice = tm.Dice(num_classes=17, average='macro', multiclass=True)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, N, _ = x.shape
        y = y.view(B, N).float()

        y1 = y - 1
        y1[y1 < 0] = 16

        output, pred = self.model(x.transpose(2, 1))
        loss = self.loss(output, y1.long())

        pred = torch.zeros_like(y)
        for item in range(len(output["pred_masks"])):
            pred_scores = output["pred_logits"][item].sigmoid()
            pred_masks = output["pred_masks"][item].sigmoid()
            pred_objectness = output["pred_scores"][item].sigmoid()
            pred_scores = torch.sqrt(pred_scores * pred_objectness)
            # max/argmax
            scores, labels = pred_scores.max(dim=-1)
            # cls threshold
            keep = scores > 0.5
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = pred_masks[keep]
            pred_masks = mask_pred_per_image > 0.5

            index = torch.where(pred_masks.sum(0) > 1)  # overlay points
            pred_masks[:, index[0].cpu().numpy()] = 0

            pred[item, :] = (pred_masks * (labels[:, None] + 1)).sum(0)

        self.train_acc(pred, y)
        self.train_miou(pred, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B, N, _ = x.shape
        y = y.view(B, N).float()

        y1 = y - 1
        y1[y1 < 0] = 16

        output, pred = self.model(x.transpose(2, 1))
        loss = self.loss(output, y1.long())

        pred = torch.zeros_like(y)
        for item in range(len(output["pred_masks"])):
            pred_scores = output["pred_logits"][item].sigmoid()
            pred_masks = output["pred_masks"][item].sigmoid()
            pred_objectness = output["pred_scores"][item].sigmoid()
            pred_scores = torch.sqrt(pred_scores * pred_objectness)
            # max/argmax
            scores, labels = pred_scores.max(dim=-1)
            # cls threshold
            keep = scores > 0.5
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = pred_masks[keep]
            pred_masks = mask_pred_per_image > 0.5

            index = torch.where(pred_masks.sum(0) > 1)  # overlay points
            pred_masks[:, index[0].cpu().numpy()] = 0

            pred[item, :] = (pred_masks * (labels[:, None] + 1)).sum(0)

        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.val_dice(pred.long(), y.long())
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dice", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        B, N, _ = x.shape
        y = y.view(B, N).float()

        y1 = y - 1
        y1[y1 < 0] = 16

        output, pred = self.model(x.transpose(2, 1))
        loss = self.loss(output, y1.long())

        pred = torch.zeros_like(y)
        for item in range(len(output["pred_masks"])):
            pred_scores = output["pred_logits"][item].sigmoid()
            pred_masks = output["pred_masks"][item].sigmoid()
            pred_objectness = output["pred_scores"][item].sigmoid()
            pred_scores = torch.sqrt(pred_scores * pred_objectness)
            # max/argmax
            scores, labels = pred_scores.max(dim=-1)
            # cls threshold
            keep = scores > 0.5
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = pred_masks[keep]
            pred_masks = mask_pred_per_image > 0.5

            index = torch.where(pred_masks.sum(0) > 1)  # overlay points
            pred_masks[:, index[0].cpu().numpy()] = 0
            pred[item, :] = (pred_masks * (labels[:, None] + 1)).sum(0)

        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.val_dice(pred.long(), y.long())
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dice", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }
