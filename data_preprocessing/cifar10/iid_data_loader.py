import os
import argparse
import time
import math
import logging
# 导入基础模块，用于文件操作、参数解析、时间处理、数学计算、日志记录等。
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler
# 导入PyTorch及其数据加载相关的模块，用于构建神经网络、处理数据集和数据增强。
from .transform import data_transforms_cifar10
# 从当前目录下的transform模块导入data_transforms_cifar10函数，该函数可能用于定义CIFAR-10数据集的变换操作。

# 这段代码定义了一个用于加载CIFAR-10数据集的函数load_iid_cifar10，它支持独立同分布（IID）数据的加载，并且可以适应分布式训练或单机训练环境。
# 下面是对代码的逐行解释：
# 整体来看，这个函数提供了一个通用的数据加载接口，支持不同的训练模式，并且可以方便地集成到分布式训练框架中。通过使用分布式采样器，
# 函数确保了在分布式训练中每个进程只处理分配给它们的数据子集。此外，函数还考虑了单机训练的情况，为每个客户端分配了完整的数据集。


def load_iid_cifar10(dataset, data_dir, partition_method, 
        partition_alpha, client_number, batch_size, rank=0, args=None):
    # 定义load_iid_cifar10函数，它接受数据集名称、数据目录、分区方法、分区参数、客户端数量、批量大小、等级（分布式训练中的标识）和参数对象。
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    # 定义CIFAR-10数据集的均值和标准差，用于数据标准化。
    # CIFAR_STD = [0.2023, 0.1994, 0.2010]

    image_size = 32
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])
    # 定义训练和测试数据集的变换操作。
    train_dataset = CIFAR10(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR10(root=data_dir, train=False,
                            transform=test_transform, download=False)
    # 加载CIFAR-10训练和测试数据集，应用上述定义的变换操作。
    if args.mode in ['distributed', 'centralized']:
        # 根据args.mode的值，处理不同的训练模式。支持分布式/集中式和单机训练模式。
        train_sampler = None
        shuffle = True
        if client_number > 1:
            train_sampler = data.distributed.DistributedSampler(
                train_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)
            shuffle = False

            # Note that test_sampler is for distributed testing to accelerate training
            test_sampler = data.distributed.DistributedSampler(
                test_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)
            # 如果处于分布式训练环境，创建分布式采样器以确保每个进程只处理数据集的一部分。

        train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4)
        test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
        # 创建全局训练和测试数据加载器。
        train_sampler = train_sampler
        train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4, sampler=train_sampler)
        test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
        # classes = ('plane', 'car', 'bird', 'cat',
        #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        class_num = 10

        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)

        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_index in range(client_number):
            # 循环为每个客户端创建数据加载器，并分配本地数据。
            train_data_local_dict[client_index] = train_dl
            test_data_local_dict[client_index] = test_dl
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            data_local_num_dict[client_index] = train_data_num // client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))
    elif args.mode == 'standalone':
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        for client_index in range(client_number):
            train_sampler = None
            shuffle = True
            if client_number > 1:
                train_sampler = data.distributed.DistributedSampler(
                    train_dataset, num_replicas=client_number, rank=client_index)
                train_sampler.set_epoch(0)
                shuffle = False

                # Note that test_sampler is for distributed testing to accelerate training
                test_sampler = data.distributed.DistributedSampler(
                    test_dataset, num_replicas=client_number, rank=client_index)
                train_sampler.set_epoch(0)


            train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=4)
            test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)

            train_sampler = train_sampler
            train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=4, sampler=train_sampler)
            test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
            # classes = ('plane', 'car', 'bird', 'cat',
            #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            class_num = 10

            train_data_num = len(train_dataset)
            test_data_num = len(test_dataset)

            train_data_local_dict[client_index] = train_dl
            test_data_local_dict[client_index] = test_dl
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            data_local_num_dict[client_index] = train_data_num // client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))
    else:
        raise NotImplementedError

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
    # 返回训练集大小、测试集大小、全局训练数据加载器、全局测试数据加载器、
    # 每个客户端的本地数据数量、每个客户端的训练和测试数据加载器、类别总数。







