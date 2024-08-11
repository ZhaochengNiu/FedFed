import logging
# 导入 Python 的日志模块，用于记录日志信息。


class PSTrainer(object):
    # 定义了 PSTrainer 类，它是一个训练器基类。
    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, device, args, model_trainer):
        # PSTrainer 类的构造函数，接收客户端索引、原始训练数据和标签、
        # 测试数据加载器、训练数据数量、测试数据数量、设备信息、参数配置和模型训练器。
        self.args = args
        # 保存参数配置。
        self.client_index = client_index
        # 保存客户端索引。
        self.train_ori_data = train_ori_data
        # 保存原始训练数据。
        self.train_ori_targets = train_ori_targets
        # 保存原始训练标签。
        self.test_dataloader = test_dataloader
        # 保存测试数据加载器。
        self.local_sample_number = train_data_num
        # 保存本地样本数量。
        self.test_data_num = test_data_num
        # 保存测试数据数量。

        logging.info(f"Initializing client: {self.client_index}"+
                    f" len(train_data) (local data num): {len(self.train_ori_data)} ")
        # 记录初始化客户端的日志信息。

        self.device = device
        # 保存设备信息。
        self.trainer = model_trainer
        # 保存模型训练器。
        # =============================================

    def update_state(self, **kwargs):
        # 定义 update_state 方法，用于更新训练器的状态。
        self.trainer.update_state(**kwargs)
        # 调用模型训练器的 update_state 方法。

    def lr_schedule(self, progress):
        # 定义 lr_schedule 方法，用于根据进度调整学习率。
        self.trainer.lr_schedule(progress)
        # 调用模型训练器的 lr_schedule 方法。

    def warmup_lr_schedule(self, iterations):
        # 定义 warmup_lr_schedule 方法，用于预热阶段的学习率调整。
        self.trainer.warmup_lr_schedule(iterations)
        # 调用模型训练器的 warmup_lr_schedule 方法。

    def set_model_params(self, weights):
        # 定义 set_model_params 方法，用于设置模型参数。
        self.trainer.set_model_params(weights)
        # 调用模型训练器的 set_model_params 方法。

    def set_grad_params(self, named_grads):
        # 定义 set_grad_params 方法，用于设置梯度参数。
        self.trainer.set_grad_params(named_grads)
        # 调用模型训练器的 set_grad_params 方法。

    def clear_grad_params(self):
        # 定义 clear_grad_params 方法，用于清除梯度参数。
        self.trainer.clear_grad_params()
        # 调用模型训练器的 clear_grad_params 方法。

    def update_model_with_grad(self):
        # 定义 update_model_with_grad 方法，用于根据梯度更新模型。
        self.trainer.update_model_with_grad()
        # 调用模型训练器的 update_model_with_grad 方法。

    def get_train_batch_data(self):
        # 定义 get_train_batch_data 方法，用于获取训练批次数据。
        try:
            # 尝试获取下一个训练批次数据。
            train_batch_data = self.train_local_iter.next()
            # 如果存在迭代器，则获取下一个批次数据。
            logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))

        except:
            # 如果迭代器已耗尽或发生异常，则重新初始化迭代器。
            self.train_local_iter = iter(self.train_local)
            train_batch_data = self.train_local_iter.next()
            # 重新初始化迭代器并获取下一个批次数据。
        return train_batch_data
        # 返回获取的训练批次数据。

    def get_model_params(self):
        # 定义 get_model_params 方法，用于获取模型参数。
        weights = self.trainer.get_model_params()
        # 调用模型训练器的 get_model_params 方法获取权重。
        model_indexes = None
        # 当前示例中，模型索引设置为 None，可能在其他上下文中使用。
        return weights, model_indexes
        # 返回模型权重和索引。

