import torch
from torch.optim import Optimizer
# 导入 PyTorch 库和优化器模块。
from .base_lr_scheduler import _LRScheduler
# 从当前目录下的 base_lr_scheduler 模块导入 _LRScheduler 抽象基类。
# 这段代码定义了一个名为 MultiStepLR 的类，它继承自 _LRScheduler 类，
# 并实现了多步学习率调度策略。以下是对代码的逐行解释：
# 整体来看，MultiStepLR 类实现了多步学习率调度策略，它允许学习率在达到预定的里程碑时按照衰减率进行衰减。
# 这种调度策略有助于在训练的不同阶段调整学习率，以促进模型的收敛。


class MultiStepLR(_LRScheduler):
    # 定义了 MultiStepLR 类，继承自 _LRScheduler。
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_decay_rate=0.1, lr_milestones=[30, 60, 90]):
        # 构造函数，初始化 MultiStepLR 类的实例。它接收优化器、基础学习率、预热周期数、迭代次数、预热类型、预热值、衰减率和里程碑列表。
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )
        # 调用基类 _LRScheduler 的构造函数，传递初始化参数。
        self.lr_decay_rate = lr_decay_rate
        # 保存学习率衰减率。
        self.lr_milestones = lr_milestones
        # 保存学习率衰减的里程碑列表。

    def get_lr(self, progress):
        # 定义 get_lr 方法，用于根据当前进度 progress 计算学习率。
        index = 0
        # 初始化一个索引计数器。
        for milestone in self.lr_milestones:
            if progress < milestone:
                break
            else:
                index += 1
        # 遍历里程碑列表，检查进度是否达到或超过每个里程碑，更新索引计数器。
        self.lr = self.base_lr * (self.lr_decay_rate**index)
        # 根据索引和衰减率计算当前学习率。
        return self.lr
        # 返回计算得到的学习率。









