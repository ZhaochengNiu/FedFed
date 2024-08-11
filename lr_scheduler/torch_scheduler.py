import torch
# 导入 PyTorch 库。

"""
    args.lr_scheduler in 
    ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    --step-size
    --lr-decay-rate
    --lr-milestones
    --lr-T-max
    --lr-eta-min
"""

# 这段代码是一个 Python 函数，用于根据提供的参数和配置创建不同的学习率调度器（Learning Rate Scheduler）。
# 以下是对代码的逐行解释：
# 整体来看，create_scheduler 函数是一个工厂函数，根据传入的参数和配置来创建和返回一个适当的学习率调度器实例。
# 它支持创建五种类型的调度器，并且可以灵活地处理不同的初始化参数。


def create_scheduler(args, optimizer):
    # 定义了一个名为 create_scheduler 的函数，它接受参数配置 args 和一个优化器 optimizer。
    if args.sched == "StepLR":
        # 如果调度器类型为 "StepLR"，则创建一个步进学习率调度器。
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.step_size, args.lr_decay_rate)
        # 使用 PyTorch 的 StepLR 调度器，设置优化器、步进大小和衰减率。
    elif args.sched == "MultiStepLR":
        # 如果调度器类型为 "MultiStepLR"，则创建一个多步学习率调度器。
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_milestones, args.lr_decay_rate)
        # 使用 PyTorch 的 MultiStepLR 调度器，设置优化器、衰减里程碑和衰减率。
    elif args.sched == "ExponentialLR":
        # 如果调度器类型为 "ExponentialLR"，则创建一个指数退火学习率调度器。
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, args.lr_decay_rate)
        # 使用 PyTorch 的 ExponentialLR 调度器，设置优化器和衰减率。
    elif args.sched == "CosineAnnealingLR":
        # 如果调度器类型为 "CosineAnnealingLR"，则创建一个余弦退火学习率调度器。
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.lr_T_max, args.lr_eta_min)
        # 使用 PyTorch 的 CosineAnnealingLR 调度器，设置优化器、最大周期和最小学习率。
    elif args.sched == "ReduceLROnPlateau":
        # 如果调度器类型为 "ReduceLROnPlateau"，则创建一个基于性能退火学习率的调度器。
        lr_scheduler = torch.optim.lr_sheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10,
            verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)
        # 使用 PyTorch 的 ReduceLROnPlateau 调度器，设置优化器和其他参数。
        # 注意这里可能有一个拼写错误：lr_sheduler 应该是 lr_scheduler。
    else:
        raise NotImplementedError
        # 如果 args.sched 不是以上任何一种，抛出 NotImplementedError 异常。

    return lr_scheduler
    # 返回创建的学习率调度器实例。






