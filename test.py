# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/22/022 13:53
@Author  : NDWX
@File    : test.py
@Software: PyCharm
"""

import glob
import torch
import code.segmentation_models_pytorch as smp
from code.utils.dataset import build_test_dataloader
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # 测试数据集
    test_dataset = [sorted(glob.glob("../data/WHU-buliding/test/image/*.tif")),
                    sorted(glob.glob("../data/WHU-buliding/test/label/*.tif"))]
    # 模型路径whubasepylkpsasangechuangxin2mszszbase4
    model = torch.load(r'../user_data/zh/whu/whubaselkconv1357catbnreluchattpsa/whubaselkconv1357scchattpsa4zh.pth')

    loss = torch.nn.CrossEntropyLoss()
    loss.__name__ = "CrossEntropyLoss"

    metrics = [
        smp.utils.metrics.IoU(activation='argmax2d'),
        smp.utils.metrics.Fscore(activation='argmax2d'),
        smp.utils.metrics.Recall(activation='argmax2d'),
        smp.utils.metrics.Precision(activation='argmax2d'),
    ]

    test_loader = build_test_dataloader(test_dataset)
    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    test_epoch.infer_vis(test_loader, evaluate=True, save=True, save_dir='../user_data/zh/whu/whubaselkconv1357catbnreluchattpsa/infer4', suffix=".png")
