import torch
# 导入 PyTorch 库。
from .steplr_scheduler import StepLR
from .multisteplr_scheduler import MultiStepLR
from .consine_lr_scheduler import CosineAnnealingLR
# 从当前目录下的不同模块导入三种不同的学习率调度器类：StepLR、MultiStepLR 和 CosineAnnealingLR。
"""
    args.lr_scheduler in 
    ["StepLR", "MultiStepLR", "CosineAnnealingLR"]
    --step-size
    --lr-decay-rate
    --lr-milestones
    --lr-T-max
    --lr-eta-min
"""


def create_scheduler(args, optimizer, **kwargs):
    """
        num_iterations is the number of iterations per epoch.
    """
    # 定义了一个名为 create_scheduler 的函数，它接受参数配置 args、一个优化器 optimizer 和额外的关键字参数 **kwargs。
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index
    # 从 kwargs 中获取客户端索引，如果没有提供，则使用 args 中的客户端索引。
    if args.sched == "no":
        lr_scheduler = None
        # 如果调度器类型为 "no"，则不创建学习率调度器，返回 None。
    elif args.sched == "StepLR":
        # 如果调度器类型为 "StepLR"，则创建一个 StepLR 调度器实例。
        lr_scheduler = StepLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_decay_rate=args.lr_decay_rate,
            step_size=args.step_size)
        # 使用提供的基础学习率、预热周期、迭代次数、预热类型、预热值和衰减率等参数初始化 StepLR。
    elif args.sched == "MultiStepLR":
        # 如果调度器类型为 "MultiStepLR"，则创建一个 MultiStepLR 调度器实例。
        lr_scheduler = MultiStepLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_decay_rate=args.lr_decay_rate,
            lr_milestones=args.lr_milestones)
        # 使用类似 StepLR 的参数，但使用里程碑（milestones）代替步长（step size）。
    elif args.sched == "CosineAnnealingLR":
        # 如果调度器类型为 "CosineAnnealingLR"，则创建一个 CosineAnnealingLR 调度器实例。
        lr_scheduler = CosineAnnealingLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_T_max=args.lr_T_max,
            lr_eta_min=args.lr_eta_min)
        # 使用基础学习率、预热周期、迭代次数、预热类型、预热值、最大周期（T_max）和最小学习率（eta_min）等参数初始化 CosineAnnealingLR。
    else:
        raise NotImplementedError
        # 如果 args.sched 不是以上任何一种，抛出 NotImplementedError 异常。
    return lr_scheduler
    # 返回创建的学习率调度器实例。








