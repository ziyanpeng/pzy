# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:45
@Author  : NDWX
@File    : metrics.py
@Software: PyCharm
"""
import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
def cal_val_iou(model, loader):
    val_iou = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output = output.argmax(1)
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou


# 计算IoU
def cal_iou(pred, mask, c=2):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        #  0.0001防止除零
        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)
