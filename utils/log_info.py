from torch.utils.tensorboard import SummaryWriter
# 函数通过 from torch.utils.tensorboard import SummaryWriter 导入了 TensorBoard 的 SummaryWriter 类。


def log_info(type: str, name: str, info, step=None, record_tool='wandb', wandb_record=False):
    '''
    type: the info type mainly include: image, scalar (tensorboard may include hist, scalars)
    name: replace the info name displayed in wandb or tensorboard
    info: info to record
    '''
    # 函数定义 log_info 接受以下参数：
    # type：信息的类型可以是 'image', 'scalar', 或 'histogram'。
    # name: 显示在 TensorBoard 或 wandb 上的信息名称。
    # info: 需要记录的信息本身。
    # step: 记录信息的步骤或迭代次数。
    # record_tool: 指定记录信息的工具，可以是 'wandb' 或 'tensorboard'。
    # wandb_record: 一个布尔值，指示是否在 Weights & Biases 上记录信息。
    if record_tool == 'wandb':
        # 首先检查 record_tool 是否为 'wandb'，如果是，则导入 wandb 库。
        import wandb
    if type == 'image':
        # 函数根据 type 参数的值，执行不同的记录操作：
        # 如果 type 是 'image'，函数会根据 record_tool 的值将图像信息记录到 TensorBoard 或 wandb。
        if record_tool == 'tensorboard':
            writer.add_image(name, info, step)
        if record_tool == 'wandb' and wandb_record:
            wandb.log({name: wandb.Image(info)})
    if type == 'scalar':
        # 如果 type 是 'scalar'，函数会记录标量信息。
        if record_tool == 'tensorboard':
            writer.add_scalar(name, info, step)
        if record_tool == 'wandb'and wandb_record:
            wandb.log({name:info})
    if type == 'histogram':
        # 如果 type 是 'histogram'，函数会记录直方图信息。
        writer.add_histogram(name, info, step)