import torch
import torch.nn as nn
from models.layer import get_graph_feature, Conv


# idx from 'key'
def get_mask_pos(idx, pos):
    batch_size, num_points, k = idx.shape
    mask = torch.sign(idx + 1)
    masked_pos = pos.reshape(batch_size * num_points, -1)[idx.contiguous().view(-1).long(), :].\
        view(batch_size, num_points, k, -1)
    masked_pos = masked_pos - pos.unsqueeze(2)
    masked_pos = torch.einsum(
        "b n k c, b n k -> b n k c", masked_pos, mask
    )
    return masked_pos.permute(0, 3, 1, 2).contiguous()


class VectorAttention(nn.Module):
    def __init__(self, channels, heads=8, k=32, attn_drop_rate=0.0, qkv_bias=True):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.k = k
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.channels_per_head = channels // heads

        self.to_q = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=qkv_bias),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.to_k = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=qkv_bias),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.to_v = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=qkv_bias),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.pos_bias = Conv(channels, channels)

        self.encoding = nn.Sequential(
            Conv(channels, self.channels_per_head),
            Conv(self.channels_per_head, self.channels_per_head)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(self.attn_drop_rate)

    def forward(self, x, pos=None):
        # x [batch, num_points, channels]
        if pos is None:
            pos = x
        batch, num_points = x.shape[0], x.shape[1]
        x = x.transpose(2, 1)
        query, key, value = self.to_q(x), self.to_k(x), self.to_v(x)
        key, idx = get_graph_feature(key, self.k, edge_function='global')
        value, _ = get_graph_feature(value, self.k, edge_function='global')
        # idx [batch, num_points, k]; key [batch, channels, num_points, k]
        masked_pos = get_mask_pos(idx, pos)
        # masked_pos [batch, 3 or channels, num_points, k]
        vector = key - query.unsqueeze(3)
        posb = self.pos_bias(masked_pos)
        vector = vector + posb
        value = (value + posb).permute(0, 2, 3, 1).contiguous()
        # vector [batch, channels, num_points, k]
        vector = self.encoding(vector)
        vector = self.drop(self.softmax(vector)).permute(0, 2, 3, 1).contiguous()
        mask = torch.sign(idx + 1)
        # vector [batch, num_points, k, c/h]; value [batch, num_points, k, channels]
        vector = torch.einsum("b n k g, b n k -> b n k g", vector, mask)
        value = value.reshape(batch, num_points, self.k, self.channels_per_head, self.heads)
        output = torch.einsum("b n k g h, b n k g -> b n g h", value, vector)
        return output.reshape(batch, num_points, self.channels)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(hidden_features, out_features, 1, bias=False),
            nn.BatchNorm1d(out_features),
            nn.SiLU()
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.transpose(2, 1)


class VectorAttentionBlock(nn.Module):
    def __init__(self, channels, heads=8, k=32, mlp_hidden_channels=512,
                 drop_path_rate=0.0, attn_drop_rate=0.0, qkv_bias=False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.k = k
        self.mlp_hidden_channels = mlp_hidden_channels
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias

        self.attn = VectorAttention(self.channels, self.heads, self.k, self.attn_drop_rate, self.qkv_bias)
        self.droppath = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()
        self.mlp = Mlp(channels, mlp_hidden_channels)

    def forward(self, x, pos=None):
        x = x + self.droppath(self.attn(x, pos))
        x = x + self.droppath(self.mlp(x))
        return x


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, in_channels):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.cv1 = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=11, padding=5, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        x = x.transpose(2, 1)
        x = x * self.cv1(x)
        x = x * self.cv2(torch.cat([torch.mean(x, 1, keepdim=True),
                                    torch.max(x, 1, keepdim=True)[0]], 1))
        return x.transpose(2, 1)


class TransformerLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""
    def __init__(self, c1, cm=512, num_heads=8, dropout=0.0, act=nn.GELU()):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.c = c1
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.embedding = nn.Linear(c1, c1 * 2)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Performs forward pass with post-normalization."""
        src = src.transpose(1, 0)
        q, k = self.embedding(src).split((self.c, self.c), 2)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src).transpose(1, 0)
