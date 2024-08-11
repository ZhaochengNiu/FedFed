import logging
from abc import ABC, abstractmethod
# 导入 Python 日志模块和抽象基类模块。
import torch
from torch.optim import Optimizer
# 导入 PyTorch 库和优化器模块。


class _LRScheduler(object):
    # 定义了一个名为 _LRScheduler 的抽象基类。
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1):
        # 构造函数，初始化学习率调度器的属性，包括优化器、基础学习率、预热周期数、迭代次数、预热类型和预热值。
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            # 检查传入的优化器是否是 PyTorch 的 Optimizer 类的一个实例。
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
            # 如果类型不匹配，抛出类型错误。
        self.optimizer = optimizer
        # 保存优化器实例。
        self.base_lr = base_lr
        # 保存基础学习率。
        self.lr = base_lr
        # 将当前学习率设置为等于基础学习率。
        self.warmup_epochs = warmup_epochs
        # 保存预热周期数。
        self.num_iterations = num_iterations
        # 保存迭代次数。
        self.lr_warmup_type = lr_warmup_type
        # 保存学习率预热类型。
        self.lr_warmup_value = lr_warmup_value
        # 保存学习率预热值。

    def update_groups(self, lr):
        # 定义 update_groups 方法，用于更新优化器中所有参数组的学习率。
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            # 遍历优化器的所有参数组，并将学习率设置为传入的 lr。

    def step(self, progress):
        # 定义 step 方法，根据进度更新学习率。
        if progress < self.warmup_epochs:
            return "warmup"
            # 如果进度小于预热周期数，则返回 "warmup"。
        else:
            lr = self.get_lr(progress)
            # 否则，调用 get_lr 方法获取新的学习率。
            self.update_groups(lr)
            # 更新优化器参数组的学习率。
            return "step"
            # 返回 "step" 表示学习率已更新。

    def warmup_step(self, iterations):
        # 定义 warmup_step 方法，用于在预热阶段更新学习率。
        if self.lr_warmup_type == "constant":
            # 如果预热类型是 "constant"，则将学习率设置为预热值。
            self.lr = self.lr_warmup_value
            # 设置学习率为预热值。
        elif self.lr_warmup_type == "gradual":
            # 如果预热类型是 "gradual"，则逐渐增加学习率。
            warmup_total_iters = self.num_iterations * self.warmup_epochs
            # 计算预热阶段的总迭代次数。
            min_lr = self.base_lr / warmup_total_iters
            # 计算最小学习率。
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            # 计算学习率间隔。
            self.lr = min_lr + lr_interval * iterations
            # 根据迭代次数计算并设置当前学习率。
        else:
            raise NotImplementedError
            # 如果预热类型不是 "constant" 或 "gradual"，则抛出未实现错误。
        self.update_groups(self.lr)
        # 更新优化器参数组的学习率。
        # logging.info(f"Scheduler::: !!!!  iterations: {iterations} self.lr: {self.lr}\n\n")


    @abstractmethod
    def get_lr(self, progress):
        """ define this function for step() using.
        """
        # 定义一个抽象方法 get_lr，它应该在子类中实现，用于根据进度获取学习率。
        pass







