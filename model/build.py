from data_preprocessing.utils.stats import get_dataset_image_size
import logging

from model.cv.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet10

from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.FL_VAE import *
# 导入数据预处理工具、日志记录库、不同的卷积神经网络架构和变分自编码器模型。

# 定义全局列表 CV_MODEL_LIST 和 RNN_MODEL_LIST，可能用于存储不同类型的模型名称：
CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]


# 定义 create_model 函数，用于根据提供的参数创建模型实例：
def create_model(args, model_name, output_dim, pretrained=False, device=None, **kwargs):
    # args：可能包含模型参数的配置对象。
    # model_name：要创建的模型名称。
    # output_dim：模型输出维度。
    # pretrained：是否使用预训练的权重。
    # device：模型运行的设备（CPU或GPU）。
    # **kwargs：其他关键字参数。
    model = None
    # 函数内部首先设置 model 为 None，记录模型名称，并根据 model_name 创建相应的模型实例：
    logging.info(f"model name: {model_name}")
    # 使用日志记录模型名称。
    if model_name in RNN_MODEL_LIST:
        # 如果模型名称在 RNN_MODEL_LIST 中，则可能有一些特定的处理，但代码中该部分被省略了（使用 pass）。
        pass
    else:
        image_size = get_dataset_image_size(args.dataset)
        # 获取数据集中图像的大小。
    if model_name == "vgg-9":
        if args.dataset in ("mnist", 'femnist', 'fmnist'):
            model = ModerateCNNMNIST(output_dim=output_dim,
                                        input_channels=args.model_input_channels)
        elif args.dataset in ("cifar10", "cifar100", "cinic10", "svhn"):
            # print("in moderate cnn")
            model = ModerateCNN(args, output_dim=output_dim)
            print("------------------params number-----------------------")
            num_params = sum(param.numel() for param in model.parameters())
            print(num_params)
    elif model_name == "resnet18_v2":
        # 使用一系列 elif 语句根据 model_name 创建并配置不同的 ResNet 模型：
        logging.info("ResNet18_v2")
        model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
    elif model_name == "resnet34_v2":
        # 根据 model_name 创建ResNet-18, ResNet-34, ResNet-50, 或 ResNet-10模型，并传递相应的参数。
        logging.info("ResNet34_v2")
        model = ResNet34(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels, device=device)
    elif model_name == "resnet50_v2":
        model = ResNet50(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
    elif model_name == "resnet10_v2":
        logging.info("ResNet10_v2")
        model = ResNet10(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels, device=device)
    else:
        # 如果 model_name 不匹配任何已知的模型名称，则抛出 NotImplementedError 异常。
        raise NotImplementedError

    return model
    # 最后，函数返回创建的模型实例。
