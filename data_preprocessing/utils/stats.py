import logging
import numpy as np
# 导入了日志记录库和NumPy库，用于数学运算。

# 这段代码提供了几个函数，用于记录数据集的统计信息和获取数据集图像的大小。以下是对代码中每个部分的详细解释：
# 整体来看，这段代码提供了数据集统计信息记录和图像大小获取的功能，有助于数据分析和处理。
# 代码中使用了日志记录来跟踪处理的进度和状态，并考虑了不同的数据集类型。

# from data_preprocessing.generative_loader import Generative_Data_Loader
# 定义了一个字典，包含了不同生成模型的图像分辨率。
generative_image_resolution_dict = {
    "style_GAN_init": 32,
    "style_GAN_init_32_c100": 32,
    "style_GAN_init_32_c62": 32,
    "style_GAN_init_64_c200": 64,
    "Gaussian_Noise": 32,
    "cifar_conv_decoder": 32,
}


def record_batch_data_stats(y_train, bs=None, num_classes=10):
    # 这个函数接受训练标签 y_train、批次大小 bs 和类别数量 num_classes，返回每个类别在批次中的数量。
    if bs is not None:
        bs = y_train.shape[0]
    # 通过遍历所有类别，并计算每个类别在 y_train 中出现的次数，来记录批次数据的类别分布。
    batch_cls_counts = {}
    for i in range(num_classes):
        num_label = (y_train == i).sum().item()
        batch_cls_counts[i] = num_label
    # logging.debug('Batch Data statistics: %s' % str(batch_cls_counts))
    return batch_cls_counts


# got it 数据的label和partition不同client的dataidx_map
# 返回net_cls_counts不同client有哪几个类，这几个类分别有，net_cls_counts中每一个key代表一个client，value还是一个dict，key是该client包含的类别，value是该类别个数
def record_net_data_stats(y_train, net_dataidx_map):
    # 这个函数接受训练标签 y_train 和客户端数据索引映射 net_dataidx_map，返回每个客户端拥有的类别及其数量。
    # 通过遍历所有客户端的数据索引，使用NumPy的 unique 函数找出每个客户端独有的类别和这些类别的数量。
    client_train_cls_counts_dict = {}
    for client_idx, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)   # 不同client有几个类，该类分别出现了几次
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_train_cls_counts_dict[client_idx] = tmp
    logging.debug('Data statistics: %s' % str(client_train_cls_counts_dict))
    return client_train_cls_counts_dict


def get_dataset_image_size(dataset):
    # 这个函数接受一个数据集名称 dataset，根据数据集的类型返回对应的图像大小。
    # 使用条件语句来判断数据集的类型，并返回相应的图像大小。
    if dataset in ["cifar10", "cifar100", "SVHN"]:
        image_size = 32
    elif dataset in ["mnist", "fmnist", "femnist", "femnist-digit"]:
        image_size = 28
    elif dataset in ["Tiny-ImageNet-200"]:
        image_size = 64
    elif dataset in generative_image_resolution_dict:
        # 如果数据集是生成模型的一种，从 generative_image_resolution_dict 字典中获取图像大小。
        image_size = generative_image_resolution_dict[dataset]
    else:
        # 如果输入的数据集名称不在已知的列表中，则记录一条日志信息并抛出 NotImplementedError 异常。
        logging.info(f"Input dataset: {dataset}, not found")
        raise NotImplementedError
    return image_size
    # 函数返回确定的图像大小。






