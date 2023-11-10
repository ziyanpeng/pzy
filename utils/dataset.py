# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:39
@Author  : NDWX
@File    : dataset.py
@Software: PyCharm
"""

import os.path
import albumentations as A
import cv2
import torch.utils.data as D
from albumentations.pytorch import ToTensorV2
import albumentations as A


# 构建dataset
class seg_dataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.val_transform = A.Compose([
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.Rotate(90, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
        self.train_transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(90, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
        self.test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.image_paths[index], -1), cv2.COLOR_BGR2RGB)
        if self.mode == "train":
            label = cv2.imread(self.label_paths[index], -1) // 255
            # label=Image.open(self.label_paths[index])//255
            transformed_data = self.train_transform(image=image, mask=label)
            image, label = transformed_data['image'], transformed_data['mask']
            return image, label.long()
        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index], -1) // 255
            # label = Image.open(self.label_paths[index]) // 255
            transformed_data = self.val_transform(image=image, mask=label)
            image, label = transformed_data['image'], transformed_data['mask']
            return image, label.long()
        elif self.mode == "test":
            label = cv2.imread(self.label_paths[index], -1) // 255
            # label = Image.open(self.label_paths[index]) // 255
            transformed_data = self.test_transform(image=image, mask=label)
            image, label = transformed_data['image'], transformed_data['mask']
            return image, label.long(), os.path.split(self.label_paths[index])[1]

    def __len__(self):
        return self.len

# 构建数据加载器
def get_dataloader(image_paths, label_paths, mode, batch_size, shuffle, num_workers, drop_last):
    dataset = seg_dataset(image_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader

# 生成dataloader
def build_dataloader(train_path, val_path, batch_size):
    train_loader = get_dataloader(train_path[0], train_path[1], "train", batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
    valid_loader = get_dataloader(val_path[0], val_path[1], "val", batch_size, shuffle=True, num_workers=0,
                                  drop_last=False)
    return train_loader, valid_loader

# 生成test dataloader
def build_test_dataloader(val_path):
    test_loader = get_dataloader(val_path[0], val_path[1], "test", 1, shuffle=False, num_workers=0,
                                 drop_last=False)
    return test_loader
