import numpy as np
import torch
from torch import nn
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
except Exception as e:
    print("Warning: Cannot import furthest point sample. Using a slower version. Did install the PointNet++ Ops Lib? ")

    def fps(xyz, npoint):
        """
        Farthest Point Sampling (FPS) algorithm for selecting a subset of points from a point cloud.

        Args:
            xyz (torch.Tensor): Input point cloud tensor of shape (B, N, C), where B is the batch size, N is the number
                                of points, and C is the number of dimensions.
            npoint (int): Number of points to select.

        Returns:
            torch.Tensor: Tensor of shape (B, npoint) containing the indices of the selected points.
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


def knn(x, k=16):
    """
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def batched_index_select(x, dim, index):
    for ii in range(1, len(x.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


def get_graph_feature(x, k=20, idx=None, pos=None, edge_function='global'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    device = x.device

    idx_org = idx
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if edge_function == 'global':
        feature = feature.permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local':
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local_global':
        feature = torch.concat([feature - x, x], dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)


class EdgeGraphConvBlock(nn.Module):
    """
    EdgeGraphConvBlock is a module that performs edge graph convolution on input features.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Can be "global", "local", or "local_global".
        k (int): Number of nearest neighbors to consider for local edge function. Default is 32.

    Raises:
        ValueError: If edge_function is not one of "global", "local", or "local_global".

    Attributes:
        edge_function (str): Type of edge function used.
        in_channels (int): Number of input channels.
        k (int): Number of nearest neighbors considered for local edge function.
        conv (nn.Sequential): Sequential module consisting of convolutional layers.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, k=32):
        super(EdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function == "local_global":
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos=None, idx=None):
        """
        Forward pass of the EdgeGraphConvBlock.

        Args:
            x (torch.Tensor): Input features.
            idx (torch.Tensor, optional): Index tensor for graph construction of shape (B, N, K), where B is the batch size. Defaults to None.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions. Default is None.

        Returns:
            torch.Tensor: Output features after edge graph convolution.
            torch.Tensor: Updated index tensor.

        """
        x_t = x.transpose(2, 1)
        if pos is None:
            pos = x
        pos_t = pos.transpose(2, 1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx, pos=pos_t)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class DilatedEdgeGraphConvBlock(nn.Module):
    """
    A block implementing a dilated edge graph convolution operation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Must be one of "global", "local", or "local_global".
        dilation_k (int): Number of nearest neighbors to consider for the dilation operation.
        k (int): Number of nearest neighbors to consider for the graph convolution operation.

    Raises:
        ValueError: If `dilation_k` is smaller than `k` or if `edge_function` is not one of the allowed values.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, dilation_k=128, k=32):
        super(DilatedEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if dilation_k < k:
            raise ValueError(f'Dilation k {dilation_k} must be larger than k {k}')
        self.dilation_k = dilation_k
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function in ["local_global"]:
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos, cd=None):
        """
        Forward pass of the dilated edge graph convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes,
                and C is the number of input channels.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions.
                Defaults to None.
            cd (torch.Tensor, optional): Pairwise distance tensor of shape (B, N, N). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_channels), where out_channels is the number of output channels.
            torch.Tensor: Index tensor of shape (B, N, K), representing the indices of the nearest neighbors.

        """
        x_t = x.transpose(2, 1)
        B, N, C = x.shape
        if cd is None:
            cd = torch.cdist(pos, pos, p=2)
        dilation_k = min(self.dilation_k, N)
        idx_l = torch.topk(cd, dilation_k, largest=False)[1].reshape(B * N, -1)
        idx_fps = fps(pos.reshape(B * N, -1)[idx_l], self.k).long()
        idx_fps = batched_index_select(idx_l, 1, idx_fps).reshape(B, N, -1)

        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx_fps)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class GraphGroupSelfAttention(nn.Module):
    """
    Graph Group Self-Attention module.

    Args:
        in_channels (int): Number of input channels.
        group_k (int, optional): Number of groups to divide the input into. Default is 32.
        num_heads (int, optional): Number of attention heads. Default is 3.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(self, in_channels, group_k=32, num_heads=3, dropout=0.1):
        super(GraphGroupSelfAttention, self).__init__()
        self.group_k = group_k
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, pos):
        """
        Forward pass of the GraphGroupSelfAttention module.
        :param x: Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes, and C is the number of input channels.
        :return: Output tensor of shape (B, N, C), representing the output of the GraphGroupSelfAttention module.
        """
        group_idx = fps(pos, self.group_k)
        groups = batched_index_select(x, 1, group_idx)  # (B, N, C) -> (B, group_k, C)
        attn_output, attn_output_weights = self.multihead_attn(x, groups, groups)
        out = attn_output + x
        return out


class BasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout.
    """
    default_act = nn.SiLU(inplace=True)
    # default_act = nn.LeakyReLU(negative_slope=0.2)
    # dropout = 0.1

    def __init__(self, in_channels, out_channels, dropout=0.0, is_out=False):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(BasicPointLayer, self).__init__()
        if is_out:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                self.default_act,
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1)


class ResidualBasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout
    with a residual connection.
    """
    default_act = nn.SiLU(inplace=True)
    # default_act = nn.LeakyReLU(negative_slope=0.2)
    # dropout = 0.1

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(ResidualBasicPointLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            self.default_act,
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            self.default_act,
            nn.Dropout(dropout)
        )
        if in_channels != out_channels:
            self.rescale = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False),
                self.default_act,
                nn.Dropout(dropout)
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)


class PointFeatureImportance(nn.Module):
    """
    Point Feature Importance module.
    """

    def __init__(self, in_channels):
        """
        Initializes the PointFeatureImportance module.
        :param in_channels: Number of input channels.
        """
        super(PointFeatureImportance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)


class STNkd(nn.Module):
    """
    STNkd module.
    """

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        # self.relu = nn.GELU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1)
        x_org = x
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].squeeze(2)
        # x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).astype(np.float32)).unsqueeze(0).repeat(batchsize, 1, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        trans_x = x.view(-1, self.k, self.k) + iden

        x_org = x_org.transpose(2, 1)
        x_org = torch.bmm(x_org, trans_x)
        return x_org


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 1), (3, 1)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 1), (3, 1)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3C2f(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(C3(self.c, self.c, 2, shortcut, g) for _ in range(n))


class RepConvN(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, (3, 1)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3=64, c4=32, expansion=1):
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, expansion), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, expansion), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=(1, 1), act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = Conv(c2, c2, k, g=c2, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=(3, 1), n=6, lightconv=False, shortcut=False, act=True):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y