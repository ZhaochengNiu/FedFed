import torch
from torch import nn
# 导入 PyTorch 库和神经网络模块 nn。

# 这段代码主要涉及图像数据的标准化处理，包括针对 CIFAR-10/100 数据集和 ImageNet 数据集的均值和标准差。以下是对代码的逐行解释：

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# 定义 CIFAR-10/100 和 ImageNet 数据集的均值和标准差。这些值用于图像的标准化处理。


def get_cifar_params(resol):
    # 定义函数 get_cifar_params，用于获取 CIFAR 数据集的均值和标准差张量。
    mean_list = []
    std_list = []
    # 初始化两个空列表，用于存储每个通道的均值和标准差。
    for i in range(3):
        # mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        # std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i]))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i]))
        # 对于三个颜色通道，创建分辨率为 resol x resol 的均值和标准差张量，并将它们添加到相应的列表中。
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)
    # 将均值和标准差列表堆叠成张量，并增加一个维度，以便它们可以用于批量数据。


def get_imagenet_params(resol):
    # 定义函数 get_imagenet_params，用于获取 ImageNet 数据集的均值和标准差张量。
    mean_list = []
    std_list = []
    # 初始化两个空列表，用于存储每个通道的均值和标准差。
    for i in range(3):
        mean_list.append(torch.full((resol, resol), IMAGENET_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), IMAGENET_STD[i], device='cuda'))
        # 对于三个颜色通道，创建分辨率为 resol x resol 的均值和标准差张量，并将它们放置在 CUDA 设备上。
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)
    # 与 CIFAR 参数获取函数类似，返回均值和标准差张量，增加了一个维度。


class CIFARNORMALIZE(nn.Module):
    # 定义 CIFARNORMALIZE 类，用于将 CIFAR 数据集的图像标准化。
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)
        # 在初始化函数中，调用 get_cifar_params 函数获取均值和标准差张量。

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        # 定义前向传播函数。
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x
        # 从输入 x 中减去均值并除以标准差，实现标准化。


class CIFARINNORMALIZE(nn.Module):
    # 类似地，定义了 CIFARINNORMALIZE、IMAGENETNORMALIZE 和 IMAGENETINNORMALIZE 类，用于执行反标准化操作。
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x


class IMAGENETNORMALIZE(nn.Module):
    # 类似地，定义了 CIFARINNORMALIZE、IMAGENETNORMALIZE 和 IMAGENETINNORMALIZE 类，用于执行反标准化操作。
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x


class IMAGENETINNORMALIZE(nn.Module):
    # 类似地，定义了 CIFARINNORMALIZE、IMAGENETNORMALIZE 和 IMAGENETINNORMALIZE 类，用于执行反标准化操作。
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x
