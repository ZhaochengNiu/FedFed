import os
import argparse
import time
import math
import logging
# 这些导入包括操作系统接口、命令行参数解析、时间处理、数学计算、日志记录等功能。
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# 从当前目录下的 transform 模块导入 data_transforms_cifar10 函数：
from torchvision.datasets import CIFAR10
# 定义load_centralized_cifar10函数，用于加载CIFAR-10数据集，并对其进行变换和封装成数据加载器
from torch.utils.data.distributed import DistributedSampler

from .transform import data_transforms_cifar10

# 这段代码定义了一个用于加载和处理CIFAR-10数据集的函数load_centralized_cifar10。以下是对代码中每个部分的解释：
# 整体来看，这个函数提供了一种便捷的方式来加载和准备CIFAR-10数据集，使其适合用于机器学习和深度学习实验。
# 通过调整参数，用户可以控制数据集的大小和变换方式。


def load_centralized_cifar10(dataset, data_dir, batch_size, 
                max_train_len=None, max_test_len=None,
                resize=32, augmentation=True,
                args=None):
    # 函数参数包括数据集名称、数据存储目录、批处理大小、最大训练/测试长度限制、图像大小调整、数据增强选项和额外参数。
    # 使用data_transforms_cifar10函数获取训练和测试数据的变换操作：
    train_transform, test_transform = data_transforms_cifar10(resize=resize, augmentation=augmentation)
    # 加载训练和测试数据集，应用相应的变换操作：
    train_dataset = CIFAR10(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR10(root=data_dir, train=False,
                            transform=test_transform, download=False)
    # 使用 CIFAR10 类从 torchvision.datasets 加载 CIFAR-10 数据集。
    if max_train_len is not None:
        # 如果提供了最大长度限制，截取数据集：
        # 根据max_train_len截取训练数据集。
        train_dataset.data = train_dataset.data[0: max_train_len]
        train_dataset.target = np.array(train_dataset.targets)[0: max_train_len]
    # 设置训练数据加载器的洗牌选项，并创建训练和测试数据加载器：
    shuffle = True
    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    # 使用DataLoader封装数据集，使其可以被神经网络训练循环使用。
    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 获取训练集和测试集的数据量以及类别数量：
    class_num = 10
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    # CIFAR-10 数据集共有10个类别。
    return train_dl, test_dl, train_data_num, test_data_num, class_num
    # 函数返回训练和测试数据加载器、训练数据量、测试数据量和类别数量。








