from __future__ import print_function
# 这行代码从 __future__ 模块导入 print_function，使得在 Python 2.x 版本中可以使用 Python 3.x 版本的打印语法。
import abc
# 导入 Python 的抽象基类模块，用于定义抽象方法。
import os
import math
# 导入 os 和 math 模块，分别用于操作系统交互和数学计算。
import numpy as np
import logging
# 导入 numpy 用于数值计算，导入 logging 用于日志记录。
import torch
import torch.utils.data
from torch import nn
# 导入 PyTorch 库及其数据模块，nn 模块包含构建神经网络所需的类。
from torch.nn import init
from torch.nn import functional as F
# 从 torch.nn 导入参数初始化模块 init 和功能模块 functional（别名为 F）。
from torch.autograd import Variable
# 导入 Variable 类，用于自动微分，但在较新的 PyTorch 版本中，Variable 已被 torch.Tensor 取代。
import pdb
import sys
# 导入 Python 调试器 pdb 和系统相关参数模块 sys。
sys.path.append('.')
sys.path.append('..')
# 将当前目录和上级目录添加到模块搜索路径中。
from utils.normalize import *
from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.cv.resnet_v2 import ResNet18
from utils.log_info import *
# 从 utils 和 model 目录导入一些自定义的模块和函数。
import torchvision.transforms as transforms
# 导入 torchvision.transforms 模块，用于图像预处理。

# 整体来看，这段代码提供了构建和使用宽残差网络和条件变分自编码器的完整框架。代码中使用了模块化设计，使得模型的各个部分可以灵活地组合和重用。
# 此外，代码还考虑了模型的可扩展性和可维护性，通过定义抽象基类和实现具体的模型类来实现。


def conv3x3(in_planes, out_planes, stride=1):
    # 定义了一个创建 3x3 卷积层的函数。
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    # 返回一个具有指定输入和输出平面数、卷积核大小、步长和填充的 Conv2d 层。

def conv_init(m):
    # 定义了一个用于初始化卷积层和批量归一化层的函数。
    classname = m.__class__.__name__
    # 获取模块的类名。
    if classname.find('Conv') != -1:
        # 检查类名中是否包含 'Conv'，即是否是卷积层。
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        # 使用 Xavier 均匀初始化方法初始化卷积层的权重。
        init.constant_(m.bias, 0)
        # 将卷积层的偏置初始化为 0。
    elif classname.find('BatchNorm') != -1:
        # 检查类名中是否包含 'BatchNorm'，即是否是批量归一化层。
        init.constant_(m.weight, 1)
        # 将批量归一化层的权重初始化为 1。
        init.constant_(m.bias, 0)
        # 将批量归一化层的偏置初始化为 0。


class wide_basic(nn.Module):
    # 定义wide_basic类，这是一个卷积神经网络的构建块，包含两个卷积层和两个批量归一化层，并在最后添加了一个shortcut连接：
    # 这个类继承自nn.Module，是构建Wide ResNet的基本单元。
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    # 定义 Wide_ResNet 类，这是一个更高级的网络结构，由多个 wide_basic 单元组成：
    # 这个类同样继承自nn.Module，用于构建宽残差网络。
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, norm=False):  # usually Wide_ResNet(28,10,0.3,10)
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.normalize = CIFARNORMALIZE(32)
        self.norm = norm

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        # 定义_wide_layer方法，用于创建多个wide_basic层的序列：
        # 这个方法是Wide_ResNet类的一个私有方法，用于重复创建wide_basic层。
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.norm:
            x = self.normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ResBlock(nn.Module):
    # 定义ResBlock类，这是一个残差块，包含两个卷积层和一个可选的批量归一化层：
    # 这个类继承自nn.Module，用于构建残差网络中的残差块。
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class AbstractAutoEncoder(nn.Module):
    # 定义AbstractAutoEncoder类，这是一个抽象基类，用于定义自编码器的接口：
    # 这个类继承自 nn.Module 和 ABCMeta，定义了自编码器必须实现的抽象方法。
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class FL_CVAE_cifar(AbstractAutoEncoder):
    # 定义FL_CVAE_cifar类，这是一个条件变分自编码器的实现，用于图像的无监督学习：
    # 这个类继承自 AbstractAutoEncoder，实现了条件变分自编码器。
    # 在 FL_CVAE_cifar 类中，定义了模型的构造函数，初始化了编码器、解码器和分类器（如果需要）：
    def __init__(self, args, d, z, device, with_classifier=True, **kwargs):
        # 构造函数接收多个参数，包括模型参数、维度、设备和是否包含分类器。
        super(FL_CVAE_cifar, self).__init__()

        # if args.dataset == 'fmnist':
        #     self.init_bn = nn.BatchNorm2d(1)
        # else:
        #     self.init_bn = nn.BatchNorm2d(3)
        self.noise_mean = args.VAE_mean
        self.noise_std1 = args.VAE_std1
        self.noise_std2 = args.VAE_std2
        self.device = device
        self.noise_type = args.noise_type
        self.encoder_former = nn.Conv2d(1, d // 2, kernel_size=4, stride=2, padding=1, bias=False) if args.dataset == 'fmnist' else \
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.decoder_last = nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False) if args.dataset == 'fmnist' else \
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False)
        if args.dataset == 'fmnist':
            self.xi_bn = nn.BatchNorm2d(1)
        else:
            self.xi_bn = nn.BatchNorm2d(3)

        self.sigmoid = nn.Sigmoid()

        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z) # 2048------>2048
        self.fc12 = nn.Linear(d * self.f ** 2, self.z) # 2048------>2048
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)  # 2048------>2048
        # constrain rx
        self.relu = nn.ReLU()

        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = ResNet18(args=args, num_classes=args.num_classes, image_size=32,model_input_channels=args.model_input_channels)

    def _add_noise(self, data, size, mean, std):
        # 定义 _add_noise 方法，用于向数据添加噪声：
        # 这个方法根据指定的噪声类型（高斯或拉普拉斯）向数据添加噪声。
        if self.noise_type == 'Gaussian':
            rand = torch.normal(mean=mean, std=std, size=size).to(self.device)
        if self.noise_type == 'Laplace':
            rand = torch.Tensor(np.random.laplace(loc=mean, scale=std, size=size)).to(self.device)
        data += rand
        return data

    def encode(self, x):
        # 定义encode方法，用于将输入数据编码为潜在空间的表示：
        # 这个方法执行编码操作，返回编码后的数据和两个全连接层的输出。
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        # 定义 reparameterize 方法，用于变分推断中的重参数化技巧：
        # 这个方法用于采样潜在空间的表示，实现重参数化技巧。
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
             return mu

    def decode(self, z):
        # 定义decode方法，用于将潜在空间的表示解码为可观测数据：
        # 这个方法执行解码操作，返回解码后的数据。
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        # 定义forward方法，用于执行模型的前向传播：
        # 这个方法执行模型的前向传播，返回重建的图像和潜在空间的表示。
        x_no_normalize = x
        bn_x = x
        x = self.encoder_former(bn_x)
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar) #+ noise* torch.randn(mu.size()).cuda()
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.decoder_last(xi)
        xi = self.xi_bn(xi)
        xi = self.sigmoid(xi)

        if self.with_classifier:
            size = xi[0].shape
            rx = x_no_normalize - xi
            rx_noise1 = self._add_noise(torch.clone(rx),size, self.noise_mean, self.noise_std1)
            rx_noise2 = self._add_noise(torch.clone(rx), size, self.noise_mean, self.noise_std2)
            data = torch.cat((rx_noise1, rx_noise2, bn_x), dim = 0)
            out = self.classifier(data)
            return out, hi, xi, mu, logvar, rx, rx_noise1, rx_noise2
        else:
            return xi

    def classifier_test(self, data):
        # 定义classifier_test和get_classifier方法，用于获取和测试分类器：
        # 这些方法用于获取分类器并执行分类操作。
        if self.with_classifier:
            out = self.classifier(data)
            return out
        else:
            raise RuntimeError('There is no Classifier')

    def get_classifier(self):
        return self.classifier
