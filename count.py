import torch
import numpy as np
import os
import json
from torch.utils.data.dataset import Dataset
from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *


def get_dataset(train_test_split=1) -> tuple[Dataset, Dataset]:
    train = Teeth3DSDataset("data/3dteethseg",
                            processed_folder=f'processed',
                            verbose=True,
                            pre_transform=PreTransform(classes=17),
                            post_transform=None, in_memory=False,
                            force_process=False, is_train=True,
                            train_test_split=train_test_split)
    test = Teeth3DSDataset("data/3dteethseg",
                           processed_folder=f'processed',
                           verbose=True,
                           pre_transform=PreTransform(classes=17),
                           post_transform=None, in_memory=False,
                           force_process=False, is_train=False,
                           train_test_split=train_test_split)
    return train, test


root_mesh_folder = "data/3dteethseg/raw"
for root, dirs, files in os.walk(root_mesh_folder):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file)) as f:
                data = json.load(f)
            labels = np.array(data["labels"])
            if np.sum(labels == 41) == 0:
                print(file)

# train_dataset, test_dataset = get_dataset(train_test_split=2)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
# val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
# i = 0
# k = torch.zeros([17])
# c = torch.zeros([1])
# j = 0
# for data in val_dataloader:
#     pos, x, label = data
    # i += 1
    # cls, count = torch.unique(label, return_counts=True)
    # c = torch.cat([c, cls])
    # if count[0]/16000 > 0.5:
    #     j += 1
    # if torch.sum(cls == 8) == 0:
    #     print(cls)
    # print(count[0]/16000)
    # k[cls.shape[0] - 1] += 1
    # print(cls, count)
    # break

# _, count_cls = torch.unique(c, return_counts=True)
# print(i)
# print(k)
# print(count_cls)
# print(k/i)
# print(count_cls/i)
# print(j/i)