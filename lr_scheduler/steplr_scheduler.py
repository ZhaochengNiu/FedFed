import torch
from torch.optim import Optimizer
# 导入 PyTorch 库和优化器模块。
from .base_lr_scheduler import _LRScheduler
# 从当前目录下的 base_lr_scheduler 模块导入 _LRScheduler 抽象基类。

# 这段代码定义了一个名为 StepLR 的类，它继承自 _LRScheduler 类，并实现了步进学习率调度策略。
# 以下是对代码的逐行解释：
# 整体来看，StepLR 类实现了步进学习率调度策略，它允许学习率在每个步进周期按照衰减率进行衰减。
# 这种调度策略有助于在训练过程中逐步降低学习率，以促进模型的稳定收敛。


class StepLR(_LRScheduler):
    # 定义了 StepLR 类，继承自 _LRScheduler。
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_decay_rate=0.97, step_size=1):
        # 构造函数，初始化 StepLR 类的实例。它接收优化器、基础学习率、预热周期数、迭代次数、预热类型、预热值、衰减率和步进大小。
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )
        # 调用基类 _LRScheduler 的构造函数，传递初始化参数。
        self.lr_decay_rate = lr_decay_rate
        # 保存学习率衰减率。
        self.step_size = step_size
        # 保存步进大小。

    def get_lr(self, progress):
        # 定义 get_lr 方法，用于根据当前进度 progress 计算学习率。
        # This aims to make a float step_size work.
        exp_num = progress / self.step_size
        # 计算进度除以步进大小的结果，这个结果将用于计算衰减的指数。
        self.lr = self.base_lr * (self.lr_decay_rate**exp_num)
        # 根据衰减率和指数计算当前学习率。
        return self.lr
        # 返回计算得到的学习率。











