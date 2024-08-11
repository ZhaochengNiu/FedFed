import os
import sys
import time
import os.path as osp
import random
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
import math
# 导入所需的模块。

# 这段代码是一个 Python 类的实现，它定义了一个名为 AdamW 的优化器，继承自 PyTorch 的 Optimizer 基类。
# AdamW 是 Adam 优化算法的一个变种，它在原有的 Adam 算法基础上增加了权重衰减（L2 正则化）的支持。以下是对代码的逐行解释：
# 整体来看，AdamW 类实现了 Adam 优化算法，并增加了权重衰减的支持。它通过重写 Optimizer 类的 __init__ 和 step 方法来实现自己的优化逻辑。
# 代码中还包含了对 AMSGrad 变体的支持，这是一种改进的 Adam 算法，可以提高其在非凸优化问题中的性能。


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    # 定义 AdamW 类，继承自 PyTorch 的 Optimizer 类。
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # 构造函数，用于初始化 AdamW 优化器。它接收以下参数：
        # params: 优化器将用于优化的参数或参数组。
        # lr: 学习率，默认值为 1e-3。
        # betas: 用于计算梯度和梯度平方的指数移动平均数的系数，默认为 (0.9, 0.999)。
        # eps: 用于提高数值稳定性的小常数，默认值为 1e-8。
        # weight_decay: 权重衰减（L2 正则化）系数，默认为 0。
        # amsgrad: 是否使用 AMSGrad 变体，默认为 False。
        if not 0.0 <= lr:
            # 检查学习率 lr 是否有效。
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        # 创建默认参数字典。
        super(AdamW, self).__init__(params, defaults)
        # 调用基类 Optimizer 的构造函数。

    def __setstate__(self, state):
        # 定义 __setstate__ 方法，用于反序列化优化器状态。
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            # 确保每个参数组都设置了 amsgrad 的默认值。

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 定义 step 方法，执行单次优化步骤。
        loss = None
        if closure is not None:
            loss = closure()
        # 如果提供了闭包（closure），则执行它并获取损失值。
        for group in self.param_groups:
            # 遍历所有参数组。
            for p in group['params']:
                # 遍历参数组中的所有参数。
                if p.grad is None:
                    continue
                # 如果参数 p 没有梯度，则跳过。
                grad = p.grad.data
                # 获取参数的梯度数据。
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # 检查梯度是否为稀疏格式，Adam 不支持稀疏梯度。
                amsgrad = group['amsgrad']
                # 获取是否使用 AMSGrad 变体的标志。
                state = self.state[p]
                # 获取参数的状态。
                # State initialization
                if len(state) == 0:
                    # 如果状态字典为空，则初始化状态。
                    state['step'] = 0
                    # 初始化步数。
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 初始化梯度的指数移动平均。
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # 初始化梯度平方的指数移动平均。
                    if amsgrad:
                        # 如果使用 AMSGrad，则初始化额外的状态。
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        # 初始化梯度平方指数移动平均的最大值。
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # 获取梯度和梯度平方的指数移动平均。
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                # 如果使用 AMSGrad，则获取梯度平方指数移动平均的最大值。
                beta1, beta2 = group['betas']
                # 获取参数组的 betas 值。
                state['step'] += 1
                # 更新步数。
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                # 计算指数衰减的一阶和二阶梯度。
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # 更新一阶梯度的指数移动平均。
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # 更新二阶梯度的指数移动平均。
                if amsgrad:
                    # 如果使用 AMSGrad，则更新二阶梯度的最大指数移动平均。
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # 计算当前的和历史的最大值。
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    # 使用最大值来计算分母。
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                # 如果不使用 AMSGrad，则直接使用当前的二阶梯度平方的指数移动平均来计算分母。
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # 计算偏差校正项。
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                # 计算步长。
                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size, torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))
                # 应用权重衰减并更新参数。
        return loss
        # 返回损失值。