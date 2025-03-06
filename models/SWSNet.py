from torch import nn
import torch
import torchmetrics as tm
from models.layer import BasicPointLayer, EdgeGraphConvBlock, DilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd, Conv, C2f, C3C2f, RepConvN, RepNCSPELAN4, HGBlock
from models.module import EdgeGraphCSPBlocku, E2f, LargeKernelRepBlock, E2fSum, get_idx, get_dilation_idx, \
    get_downsample_idx, get_downsample_dilated_idx
from models.utils import fuse_conv_and_bn
import lightning as L
from models.loss import DiceLoss, LovaszLoss
from models.attention import VectorAttentionBlock, SpatialAttention, TransformerLayer


class LitSWSNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SWSNet(num_classes=17)
        # self.model = SWSNetTwoStream(num_classes=17)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.04)
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_dice = tm.Dice(num_classes=17, average='macro', multiclass=True)

        self.lrf = 0.01
        self.lf = lambda x: (1 - x / self.trainer.max_epochs) * (1.0 - self.lrf) + self.lrf
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        # loss = self.loss(pred, y.long()) * 0.4 + self.loss1(pred, y.long()) * 0.6
        self.train_acc(pred, y)
        self.train_miou(pred, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.val_dice(pred, y.long())
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dice", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss 
    
    def test_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.view(B, N).float()
        pred = self.model(x, pos)
        pred = pred.transpose(2, 1)
        loss = self.loss(pred, y.long())
        self.val_acc(pred, y)
        self.val_miou(pred, y)
        self.val_dice(pred, y.long())
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dice", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)

    def predict_labels(self, data):
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            with torch.no_grad():
                pos, x, y = data
                pos = pos.unsqueeze(0).to(self.device)
                x = x.unsqueeze(0).to(self.device)
                B, N, C = x.shape
                x = x.float()
                y = y.view(B, N).float()
                pred = self.model(x, pos)
                pred = pred.transpose(2, 1)
                pred = torch.argmax(pred, dim=1)
                return pred.squeeze()

    def configure_optimizers(self):
        g = [], [], []
        lr = 1e-3
        momentum = 0.9
        weight_decay = 1e-5
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm2d' in k)
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if isinstance(module, bn):
                    g[1].append(param)
                elif 'gamma' in fullname:
                    g[0].append(param)
                else:
                    g[0].append(param)
        optimizer = torch.optim.Adam(g[0], lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        # optimizer.add_param_group({'params': g[2], 'lr': 5e-4, 'weight_decay': 0.0})

        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)

        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5, verbose=True)  # original
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1, verbose=True)
        # sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf, verbose=True)
        # sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 76], gamma=0.1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }

    @torch.no_grad()
    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, LargeKernelRepBlock):
                m.fuse()
                m.forward = m.forward_fuse
            if isinstance(m, RepConvN):
                m.fuse()
                m.forward = m.forward_fuse
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        return self


class SWSNet(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, k=16, sample_rate=None, dilated_rate=None):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        if sample_rate is None:
            # sample_rate = [1, 1, 1]
            # sample_rate = [1, 2, 4]
            # sample_rate = [2, 4, 8]
            sample_rate = [4, 8, 16]
            # sample_rate = [16, 32, 64]
            # sample_rate = [8, 16, 32]
            # sample_rate = [32, 64, 128]
            # sample_rate = [64, 128, 256]
        if dilated_rate is None:
            dilated_rate = [8, 8, 8]
            # dilated_rate = [16, 16, 16]
            # dilated_rate = [32, 32, 32]
            # dilated_rate = [32, 64, 128]
            # dilated_rate = [64, 128, 256]
            # dilated_rate = [16, 32, 64]
            # dilated_rate = [8, 16, 32]
            # dilated_rate = [4, 4, 4]
            # dilated_rate = [2, 2, 2]
            # dilated_rate = [1, 1, 1]
        self.sample_rate = sample_rate
        self.dilated_rate = dilated_rate

        self.stnkd = STNkd(k=feature_dim)
        self.e_local = E2f(feature_dim, 128, 256, self.k, 2)#, block=Conv)
        self.e0 = E2f(256, 256, 256, self.k, 2)#, block=Conv)
        self.e1 = E2f(256, 256, 256, self.k, 2)#, block=Conv)
        self.e2 = E2f(256, 256, 512, self.k, 2)#, block=Conv)

        self.attention = SpatialAttention(in_channels=512)
        self.res_block1 = ResidualBasicPointLayer(in_channels=512, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

    def forward(self, x, pos):
        cd = torch.cdist(pos, pos)
        idx = get_idx(self.k, cd=cd)
        # idx = get_downsample_dilated_idx(self.k, 1, 16, pos)
        sample_idx = []
        for i in range(len(self.sample_rate)):
            sample_idx.append(get_downsample_dilated_idx(self.k, self.sample_rate[i], self.dilated_rate[i], pos))
        x = self.stnkd(x)
        x = self.e_local(x, idx)
        x = self.e0(x, sample_idx[0])
        x = self.e1(x, sample_idx[1])
        x = self.e2(x, sample_idx[2])
        # x = self.e0(x, idx)
        # x = self.e1(x, idx)
        # x = self.e2(x, idx)
        x = self.attention(x)
        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.out(x)
        return x


class SWSNetTwoStream(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24, k=16, sample_rate=None, dilated_rate=None):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        if sample_rate is None:
            sample_rate = [4, 8, 16]
        if dilated_rate is None:
            dilated_rate = [8, 8, 8]
        self.sample_rate = sample_rate
        self.dilated_rate = dilated_rate

        self.stnkd_c = STNkd(k=feature_dim // 2)
        self.stnkd_n = STNkd(k=feature_dim // 2)

        self.c_local = E2f(feature_dim // 2, 128, 128, self.k, 2)
        self.n_local = E2f(feature_dim // 2, 128, 128, self.k, 2)

        self.c0 = E2f(128, 128, 128, self.k, 2)
        self.c1 = E2f(128, 128, 128, self.k, 2)
        self.c2 = E2f(128, 128, 256, self.k, 2)

        self.n0 = E2f(128, 128, 128, self.k, 2)
        self.n1 = E2f(128, 128, 128, self.k, 2)
        self.n2 = E2f(128, 128, 256, self.k, 2)

        self.c_attention = SpatialAttention(in_channels=256)
        self.n_attention = SpatialAttention(in_channels=256)
        # self.attention = SpatialAttention(in_channels=512)
        self.res_block1 = ResidualBasicPointLayer(in_channels=512, out_channels=512, hidden_channels=512)
        self.res_block2 = ResidualBasicPointLayer(in_channels=512, out_channels=256, hidden_channels=256)

        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

    def forward(self, x, pos):
        cd = torch.cdist(pos, pos)
        idx = get_idx(self.k, cd=cd)
        sample_idx = []
        for i in range(len(self.sample_rate)):
            sample_idx.append(get_downsample_dilated_idx(self.k, self.sample_rate[i], self.dilated_rate[i], pos))

        c, n = x.chunk(2, dim=2)

        c = self.stnkd_c(c)
        c = self.c_local(c, idx)
        c = self.c0(c, sample_idx[0])
        c = self.c1(c, sample_idx[1])
        c = self.c2(c, sample_idx[2])
        c = self.c_attention(c)

        n = self.stnkd_n(n)
        n = self.n_local(n, idx)
        n = self.n0(n, sample_idx[0])
        n = self.n1(n, sample_idx[1])
        n = self.n2(n, sample_idx[2])
        n = self.n_attention(n)

        x = torch.concat([c, n], dim=2)
        # x = self.attention(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out(x)
        return x
