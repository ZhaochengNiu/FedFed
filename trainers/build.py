import os
import sys
# 导入 Python 的 os 和 sys 模块，这些模块提供了与操作系统交互的功能。

# 这段代码定义了一个名为 create_trainer 的函数，用于创建并返回一个训练器实例。以下是对代码的逐行解释：
# 整体来看，create_trainer 函数是一个工厂函数，用于根据提供的参数和配置创建训练器实例。
# 它负责初始化模型训练所需的优化器、损失函数、学习率调度器，并最终创建训练器对象。这种设计模式使得训练器的创建过程灵活且易于扩展。

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
# 将当前目录的上一级目录添加到模块搜索路径 sys.path 的开头，这样 Python 就可以导入该目录下的模块。

from .normal_trainer import NormalTrainer
# 从当前目录下的 normal_trainer 模块导入 NormalTrainer 类。
from optim.build import create_optimizer
from loss_fn.build import create_loss
from lr_scheduler.build import create_scheduler
# 从 optim、loss_fn 和 lr_scheduler 模块中导入 create_optimizer、create_loss 和 create_scheduler 函数。


def create_trainer(args, device, model=None, **kwargs):
    # 定义了一个名为 create_trainer 的函数，它接受以下参数：
    # args：包含训练参数的配置对象。
    # device：模型运行的设备，例如 CPU 或 GPU。
    # model：要训练的模型，如果提供了 params，则此参数将被忽略。
    # **kwargs：额外的关键字参数，可用于传递其他配置。
    params = None
    # 初始化 params 为 None，这是一个备用参数，如果提供了具体的参数，则使用这些参数而不是模型的参数。
    optimizer = create_optimizer(args, model, params=params, **kwargs)
    # 调用 create_optimizer 函数创建优化器，使用 args、model 和 params 作为参数。
    criterion = create_loss(args, device, **kwargs)
    # 调用 create_loss 函数创建损失函数，使用 args 和 device 作为参数。
    lr_scheduler = create_scheduler(args, optimizer, **kwargs)   # no for FedAvg
    # 调用 create_scheduler 函数创建学习率调度器，使用 args 和 optimizer 作为参数。
    model_trainer = NormalTrainer(model, device, criterion, optimizer, lr_scheduler, args, **kwargs)
    # 创建 NormalTrainer 类的实例，它是一个普通的训练器，用于训练模型。
    # 将模型、设备、损失函数、优化器、学习率调度器和 args 作为参数传递。
    return model_trainer
    # 返回创建的 model_trainer 实例。










