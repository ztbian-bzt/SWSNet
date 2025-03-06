import time

import numpy as np
import torch
import torch.nn as nn
from models.utils import fuse_conv_and_bn
from models.layer import batched_index_select, C2f, BasicPointLayer, Conv, C3C2f, RepNCSPELAN4, HGBlock
from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
from pointnet2_ops.pointnet2_utils import ball_query


class EdgeGraphCSPBlocku(nn.Module):
    def __init__(self, in_channels, out_channels, edge_function, k=32, block=C2f):
        super(EdgeGraphCSPBlocku, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed.')
        if edge_function == "local_global":
            self.hidden_channels = self.in_channels * 2
        else:
            self.hidden_channels = self.in_channels
        if block in [C2f, C3C2f]:
            self.conv = block(self.hidden_channels, out_channels, 2, True)
        elif block in [HGBlock]:
            self.conv = block(self.hidden_channels, out_channels//2, out_channels, (3, 1), 6, True, True)
        elif block in [RepNCSPELAN4]:
            self.conv = block(self.hidden_channels, out_channels)
        elif block in [Conv]:
            self.conv = nn.Sequential(block(self.hidden_channels, out_channels, (1, 1)),
                                      block(out_channels, out_channels, (1, 1)))

    def forward(self, x, idx):
        out, idx = get_graph_feature(x, edge_function=self.edge_function, k=self.k, idx=idx)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out


def get_dilation_idx(k, dilated_k, pos, cd=None):
    batch, point_num, _ = pos.shape
    if cd is None:
        cd = torch.cdist(pos, pos, p=2)
    idx_l = torch.topk(cd, dilated_k, largest=False)[1].reshape(batch * point_num, -1)
    idx_fps = fps(pos.reshape(batch * point_num, -1)[idx_l], k).long()
    idx_fps = batched_index_select(idx_l, 1, idx_fps).reshape(batch, point_num, -1)
    return idx_fps


def get_downsample_idx(k, downsample_rate, pos):
    batch, point_num, _ = pos.shape
    device = pos.device
    downsample_point_num = int(point_num // downsample_rate)
    idx_fps = fps(pos, downsample_point_num)
    idx_origin = idx_fps
    idx_base = torch.arange(0, batch, device=device).unsqueeze(1) * point_num
    idx_fps = idx_fps + idx_base
    new_pos = pos.reshape(batch * point_num, -1)[idx_fps, :]
    cd = torch.cdist(pos, new_pos, p=2)
    idx = torch.topk(cd, k=k, dim=-1, largest=False)[1]
    idx = idx + idx_base.unsqueeze(1) // downsample_rate
    idx = idx_origin.reshape(batch * downsample_point_num, -1)[idx].squeeze(3)
    return idx


def get_downsample_dilated_idx(k, downsample_rate, dilated_rate, pos, out=False):
    batch, point_num, _ = pos.shape
    device = pos.device
    assert downsample_rate >= 1 and dilated_rate >= 1
    if downsample_rate > 1:
        downsample_point_num = int(point_num // downsample_rate)
        idx_fps = fps(pos, downsample_point_num).long()
        idx_origin = idx_fps
        idx_base = torch.arange(0, batch, device=device).unsqueeze(1) * point_num
        idx_fps = idx_fps + idx_base
        new_pos = pos.reshape(batch * point_num, -1)[idx_fps, :]
        cd = torch.cdist(pos, new_pos, p=2)

        idx_l = torch.topk(cd, k=k * dilated_rate, dim=-1, largest=False)[1]
        idx_l = idx_l + idx_base.unsqueeze(1) // downsample_rate
        idx_l = idx_origin.reshape(batch * downsample_point_num, -1)[idx_l].squeeze(3)
        if dilated_rate == 1:
            return idx_l
        idx_l = idx_l.reshape(batch * point_num, -1)
    else:
        cd = torch.cdist(pos, pos, p=2)
        idx_l = torch.topk(cd, k=k * dilated_rate, largest=False)[1].reshape(batch * point_num, -1)
    # start = time.time()
    idx = fps(pos.reshape(batch * point_num, -1)[idx_l], k).long()
    idx = batched_index_select(idx_l, 1, idx).reshape(batch, point_num, -1)
    # print(time.time() - start)
    # for visualization
    if out:
        try:
            return idx_fps, idx_l.reshape(batch, point_num, -1), idx
        except NameError:
            return idx_l.reshape(batch, point_num, -1), idx

    return idx


def get_idx(k, pos=None, cd=None):
    if cd is None:
        assert pos is not None
        cd = torch.cdist(pos, pos, p=2)
    idx = cd.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def get_graph_feature(x, idx, k=32, edge_function='local_global'):
    batch_size, num_points, num_dims = x.shape
    device = x.device

    idx_org = idx
    idx_base = torch.arange(0, batch_size, device=device).unsqueeze(1).unsqueeze(1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(-1)
    feature = x.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)

    if edge_function == 'global':
        feature = feature.permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local':
        x = x.unsqueeze(2)
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local_global':
        x = x.unsqueeze(2).repeat(1, 1, k, 1)
        feature = torch.concat([feature - x, x], dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_org  # (batch_size, 2*num_dims/num_dims, num_points, k)


class E2f(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, n=3, e=0.5, block=C2f,
                 edge_function="local_global"):
        super().__init__()
        self.hidden_channels = int(hidden_channels * e)
        self.n = n
        if in_channels != hidden_channels:
            self.cv1 = BasicPointLayer(in_channels, hidden_channels)
        else:
            self.cv1 = nn.Identity()
        self.cv2 = BasicPointLayer((2 + n) * self.hidden_channels, out_channels)
        self.block = nn.ModuleList(EdgeGraphCSPBlocku(self.hidden_channels, self.hidden_channels,
                                                      edge_function, k, block) for _ in range(n))

    def forward(self, x, idx):
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1], idx) for m in self.block)
        return self.cv2(torch.cat(y, 2))


class LargeKernelRepBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = Conv(channels, channels, (7, 1), 1, (3, 0), g=1, act=False)
        self.conv1 = Conv(channels, channels, (3, 1), 1, (1, 0), g=1, act=False)
        self.dim = channels
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        del self.conv.bn
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        del self.conv1.bn

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [0, 0, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class ConditionalIdentity(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, (3, 1), g=1),
            Conv(c1, 2 * c_, 1),
            LargeKernelRepBlock(2 * c_) if lk else Conv(2 * c_, 2 * c_, (3, 1), g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, (3, 1), g=1),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


class EdgeGraphCSPBlockuSum(EdgeGraphCSPBlocku):
    def __init__(self, in_channels, out_channels, edge_function, k=32, block=C2f):
        super(EdgeGraphCSPBlockuSum, self).__init__(in_channels, out_channels, edge_function, k, block)

    def forward(self, x, idx):
        out, idx = get_graph_feature(x, edge_function=self.edge_function, k=self.k, idx=idx)
        out = self.conv(out)
        out = out.sum(dim=-1, keepdim=False)
        out = out.transpose(2, 1)
        return out


class E2fSum(E2f):
    def __init__(self, in_channels, hidden_channels, out_channels, k, n=3, e=0.5, block=C2f,
                 edge_function="local_global"):
        super().__init__(in_channels, hidden_channels, out_channels, k, n, e, block)
        self.block = nn.ModuleList(EdgeGraphCSPBlockuSum(self.hidden_channels, self.hidden_channels,
                                                         edge_function, k, block) for _ in range(n))





