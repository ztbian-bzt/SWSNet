import torch
import numpy as np
import torch.nn as nn
from models.SWSNet import LitSWSNet, SWSNet
from models.layer import Conv, STNkd
from models.utils import fuse_conv_and_bn
from models.module import EdgeGraphCSPBlocku, LargeKernelRepBlock, \
    get_downsample_idx, get_dilation_idx, get_downsample_dilated_idx, get_idx
from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork
from models.THISNet import THISNet
from models.DBGANet import DBGANet
from models.loss import Cal_Loss
from thop import profile
from models.pointnet2 import Pointnet2
from models.DGCNN import DGCNN
from models.meshsegnet import MeshSegNet
from scipy.spatial import distance_matrix
from models.TSGCNet import TSGCNet
import time

# Create the model
# model = TestToothSegmentationNetwork(num_classes=17, feature_dim=24).cuda()
# model = LitSWSNet.load_from_checkpoint('/home/ztbian/project/Medical/dilated_tooth_seg_net-main/logs/t1213/3/epoch=81-step=49200.ckpt').model.cuda()
# model = DilatedToothSegmentationNetwork().cuda()
# model = DBGANet(num_classes=17, num_neighbor=32).cuda()
# model = THISNet(k=32, in_channels=24, output_channels=17).cuda()
# model = Pointnet2(17).cuda()
# model = DGCNN(17).cuda()
# model = MeshSegNet(17, 15).cuda()
# model = TSGCNet(32, 12, 17).cuda()
# # dummy input

pos = torch.rand(1, 16000, 3).cuda()  # xyz coordinates of the points. Shape: (batch_size, num_points, 3)
x = torch.rand(1, 16000, 24).cuda() # features of the points. Shape: (batch_size, num_points, feature_dim)
# start = time.time()
# y = model(x, A_L, A_S)
# y = model(x, pos)
# get_downsample_dilated_idx(16, 1, 128, pos)
# print(time.time() - start)

# flops, params = profile(model, inputs=(torch.rand(1, 16000, 24).cuda(), torch.rand(1, 16000, 3).cuda()))
# flops, params = profile(model, inputs=(torch.rand(1, 24, 16000).cuda(),))
# print(flops/1E9, params)


# for module_name, module in model.named_modules():
#     for param_name, param in module.named_parameters(recurse=False):
#         fullname = f'{module_name}.{param_name}' if module_name else param_name
#         if 'gamma' in fullname:
#             print(param_name, param)


sample_rate = [4, 8, 16]
dilated_rate = [8, 8, 8]
sample_idx = []
start = time.time()
for i in range(len(sample_rate)):
    sample_idx.append(get_downsample_dilated_idx(16, sample_rate[i], dilated_rate[i], pos))
print(time.time() - start)

sample_rate = [4, 8, 16]
dilated_rate = [8, 8, 8]
start = time.time()
for i in range(len(sample_rate)):
    sample_idx.append(get_downsample_dilated_idx(16, sample_rate[i], dilated_rate[i], pos))
print(time.time() - start)

sample_rate = [1, 1, 1]
dilated_rate = [32, 64, 128]
start = time.time()
for i in range(len(sample_rate)):
    sample_idx.append(get_downsample_dilated_idx(16, sample_rate[i], dilated_rate[i], pos))
print(time.time() - start)