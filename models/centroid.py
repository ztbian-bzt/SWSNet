import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def torch_normalize(p):
    l = p.shape[0]
    centroid = torch.mean(p, axis=0)
    p = p - centroid.unsqueeze(1)
    m = torch.max(torch.sqrt(torch.sum(p**2, axis=1)), axis=1,keepdim=True)[0]
    p = p/m.unsqueeze(1)
    return p


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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
    batch_indices = torch.arange(0, B, device=device, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :].contiguous()
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(0, B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(0, N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        # new_xyz = index_points(xyz, fps(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.concat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
 
    
class centroid(nn.Module):
    def __init__(self):
        super(centroid, self).__init__()
        input_feauture_num = 6
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], input_feauture_num, [[input_feauture_num, 32, 32], [6, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32+32, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(16, [0.2, 0.4], [16, 32], 128+128, [[256, 196, 256], [256, 196, 256]])

        self.dist_conv_1 = nn.Conv1d(512,128, 1)
        self.dist_bn_1 = nn.BatchNorm1d(128)
        self.dist_conv_2 = nn.Conv1d(128,64, 1)
        self.dist_bn_2 = nn.BatchNorm1d(64)

        self.dist_result_conv = nn.Conv1d(64, 3, 1)
        self.dist_result_conv1 = nn.Conv1d(64, 1, 1)

        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        dist_feature = F.relu(self.dist_bn_1(self.dist_conv_1(l3_points)))
        dist_feature = F.relu(self.dist_bn_2(self.dist_conv_2(dist_feature)))
        dist_result = self.dist_result_conv(dist_feature)
        class_result = self.dist_result_conv1(dist_feature)
#         print("1:",class_result.shape,dist_result.shape)
        
        class_result = class_result.transpose(2,1).contiguous()
#         print("1:",class_result.shape)
#         class_result = torch.nn.Softmax(dim=-1)(class_result.view(-1, 2))
#         print("2:",class_result.shape)
#         class_result = class_result.view(B, 14, 2)
        class_result = class_result.squeeze(2)
#         print("3:",class_result.shape)
        
        dist_feature = dist_feature.view(B, -1)
        dist_feature = F.relu(self.fc1(dist_feature))
        class_result = F.relu(self.fc2(dist_feature))
        
        weight = self.fc3(dist_feature)
        weight = F.softmax(weight,dim=1)

        return dist_result, class_result, weight

