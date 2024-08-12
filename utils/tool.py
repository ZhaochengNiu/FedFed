import torchvision
from utils.log_info import *
# 导入 torchvision 库，它提供了处理图像和视频的实用工具。
# 从 utils.log_info 模块导入了 log_info 函数，这个函数可能用于记录信息到日志或可视化工具（如 TensorBoard 或 Weights & Biases）。

# def train_reconst_images(args, client_index, data, mode,step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', 'client_{client_index}_train_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode), \
#              grid_X,step, tool=args.record_tool)
#
# def generate_reconst_images(args, client_index, data, mode, step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', "client_{client_index}_generate_{mode}.jpg".format(client_index=client_index, mode=mode), \
#              grid_X, tool=args.record_tool)
#
# def generate_reconst_images(args, client_index, data, mode, step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', "client_{client_index}_test_{mode}.jpg".format(client_index=client_index, mode=mode), \
#              grid_X, step, tool=args.record_tool)
#

# 这段代码定义了四个函数，用于创建和记录图像数据的可视化。
# 每个函数都使用 torchvision.utils.make_grid 来生成图像网格，然后使用 log_info 函数记录这些图像。下面是对每个函数的逐行解释：
# 这些函数的目的是帮助研究人员或开发人员可视化和记录模型训练、生成或测试过程中的图像数据。
# 通过 log_info 函数，这些图像可以被发送到指定的记录工具，例如 TensorBoard 或 Weights & Biases，以便于监控和分析。
# 注意，wandb_record 参数只在 log_info 函数中使用，这里默认为 False。
# 如果需要在 Weights & Biases 上记录，需要将其设置为 True 并确保 record_tool 参数设置为 'wandb'。


def train_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    # 定义一个名为 train_reconst_images 的函数，用于记录训练过程中的图像数据。
    # 函数接受 client_index（客户端索引）、data（图像数据）、mode（模式）、step（步骤或迭代次数）、
    # record_tool（记录工具）、wandb_record（是否在 Weights & Biases 上记录）和 size（图像大小）作为参数。
    grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
    # 使用 make_grid 函数从 data 中选择前64张图像，每行8张，设置填充，并规范化像素值。
    log_info('image', 'client_{client_index}_train_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)
    # 调用 log_info 函数记录图像，使用格式化字符串生成日志信息的名称。


def generate_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    # 定义一个名为 generate_reconst_images 的函数，用于记录生成过程中的图像数据。
    grid_X = torchvision.utils.make_grid(data[:16], nrow=8, padding=2, normalize=True)
    # 这里选择前 16 张图像来创建网格，其他参数与 train_reconst_images 函数相同。
    log_info('image', 'client_{client_index}_generate_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)


def test_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    # 定义一个名为 test_reconst_images 的函数，用于记录测试过程中的图像数据。
    grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
    log_info('image', 'client_{client_index}_test_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)


def Share_constuct_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    # 定义一个名为 Share_constuct_images 的函数，用于记录共享或全局构造的图像数据。
    grid_X = torchvision.utils.make_grid(data, nrow=8, padding=2, normalize=True)
    # 与之前不同，这个函数对所有图像数据使用 make_grid，而不是只选择一部分。
    log_info('image', 'client_{client_index}_global_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)
    # 记录图像，使用 "global" 来表示这是全局或共享的图像数据。
