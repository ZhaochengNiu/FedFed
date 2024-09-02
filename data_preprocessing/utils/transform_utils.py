from __future__ import print_function
# 这行代码从 __future__ 模块导入 print_function，这样做可以确保在Python 2.x版本中使用Python 3.x的打印功能。这是为了向后兼容。
import math
# 导入Python的数学库，提供各种数学运算功能。
import numpy as np
# 导入NumPy库，并将其别名设置为 np，
# 这是一个广泛使用的科学计算库，提供多维数组对象、派生对象（如掩码数组、矩阵）以及用于快速操作数组的各种例程。
import torch
# 导入PyTorch库，这是一个流行的开源机器学习库，用于深度学习，提供自动微分、GPU加速等特性。
import torch.optim as optim
# 从PyTorch库导入 optim 模块，它包含多种优化算法，用于神经网络的参数更新。

# 这个类可能用于数据增强，特别是在训练深度学习模型时，通过创建同一图像的两个不同裁剪版本来增加数据集的多样性。
# 这种方法可以帮助模型学习到更鲁棒的特征。


class TwoCropTransform:
    # 定义了一个名为 TwoCropTransform 的新类。
    """Create two crops of the same image"""
    # 这是 TwoCropTransform 类的文档字符串，说明这个类的用途是创建同一图像的两个裁剪版本。
    def __init__(self, transform):
        # 定义了类的构造函数 __init__，它接受一个参数 transform。
        self.transform = transform
        # 在构造函数内部，将传入的 transform 参数赋值给实例变量 self.transform。这个变量将存储用于后续调用的变换。

    def __call__(self, x):
        # 定义了 __call__ 方法，它允许类的实例像函数一样被调用。这个方法接受一个参数 x。
        return [self.transform(x), self.transform(x)]
        # __call__ 方法的返回值是一个列表，包含两次调用 self.transform 变换后的结果。
        # 这意味着它将同一个图像 x 通过 self.transform 变换两次，并将结果作为列表返回。





