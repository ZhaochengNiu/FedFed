import torch
from torch.optim import Optimizer
# 导入 PyTorch 库和优化器模块。
import numpy as np
# 导入 NumPy 库，用于数学运算。
from .base_lr_scheduler import _LRScheduler
# 从相对路径 . 导入 base_lr_scheduler 模块中的 _LRScheduler 抽象基类。

# 整体来看，CosineAnnealingLR 类实现了余弦退火学习率调度策略，它在预热期结束后，按照余弦函数的形式逐渐降低学习率，
# 直至达到最小学习率或完成一个周期。这种调度策略有助于模型在训练过程中更平稳地收敛。


class CosineAnnealingLR(_LRScheduler):
    # 定义了 CosineAnnealingLR 类，继承自 _LRScheduler。
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0, 
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_T_max=100, lr_eta_min=0):
        # 构造函数，初始化 CosineAnnealingLR 类的实例。
        # 它接收优化器、基础学习率、预热周期数、迭代次数、预热类型、预热值、最大周期 T_max 和最小学习率 eta_min。
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )
        # 调用基类 _LRScheduler 的构造函数，传递初始化参数。
        self.lr_T_max = lr_T_max
        # 保存最大周期 T_max 参数。
        self.lr_eta_min = lr_eta_min
        # 保存最小学习率 eta_min 参数。

    def get_lr(self, progress):
        # 定义 get_lr 方法，用于根据当前进度 progress 计算学习率。
        e = progress - self.warmup_epochs
        # 计算预热后经过的周期数。
        es = self.lr_T_max - self.warmup_epochs
        # 计算预热后到 T_max 的周期数。
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        # 根据余弦退火公式计算学习率。
        self.lr = lr
        # 将计算得到的学习率保存为实例变量。
        return self.lr
        # 返回计算得到的学习率。





