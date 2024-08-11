import logging
import copy
# 导入Python标准库中的logging和copy模块，分别用于日志记录和对象复制。
from algorithms_standalone.basePS.client import Client
from model.build import create_model
# 从相应的模块导入 Client 基类和 create_model 函数。


class FedAVGClient(Client):
    # 定义了FedAVGClient类，继承自Client类。
    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num):
        # 这是 FedAVGClient 类的构造函数，它接收多个参数来初始化客户端的属性。
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                test_data_num,  train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num)
        # 调用父类 Client 的构造函数，传递所有必要的参数。
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations
        # 初始化一个字典来存储每个客户端的本地迭代次数，并将当前客户端的索引作为键。
        self.global_epochs_per_round = self.args.global_epochs_per_round
        # 设置客户端每轮通信的全局训练周期数。
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round
        # 初始化一个字典来存储每个客户端每轮通信的本地训练周期数。

        #========================SCAFFOLD=====================#
        if self.args.scaffold:
            # 检查是否启用了 scaffold 参数。
            self.c_model_local = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
                # 如果启用了 scaffold，则创建一个本地客户端模型并初始化其参数。
            for name, params in self.c_model_local.named_parameters():
                params.data = params.data*0
                # 将新创建的本地客户端模型的所有参数置零。

    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        # 定义了一个名为 lr_schedule 的方法，用于根据迭代次数和预热周期来调整学习率。
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)

    def test(self, epoch):
        # 定义了一个名为 test 的方法，用于在给定的周期 epoch 上测试客户端模型并返回平均准确率。
        acc_avg = self.trainer.test(epoch, self.test_dataloader, self.device)
        return acc_avg

    def fedavg_train(self, share_data1, share_data2, share_y,round_idx=None, global_other_params=None, 
                    shared_params_for_simulation=None,
                     **kwargs):
        # 定义了一个名为 fedavg_train 的方法，实现了 FedAVG 算法的客户端训练逻辑。
        client_other_params = {}
        train_kwargs = {}
        # 在 fedavg_train 方法中，初始化客户端其他参数和训练参数的字典。
        # ========================SCAFFOLD/FedProx=====================#
        if self.args.fedprox or self.args.scaffold:
            # 检查是否启用了 fedprox 或 scaffold 参数。
            previous_model = copy.deepcopy(self.trainer.get_model_params())
            # 如果启用了 fedprox 或 scaffold，则复制上一轮的模型参数。
            train_kwargs['previous_model'] = previous_model
        # 如果启用了 FedProx 或 SCAFFOLD，复制上一轮的模型参数。
        # ========================SCAFFOLD/FedProx=====================#

        # ========================SCAFFOLD=====================#
        if self.args.scaffold:
            # 再次检查是否启用了 scaffold 参数。
            c_model_global = global_other_params["c_model_global"]
            # 如果启用了 scaffold，则从全局其他参数中获取全局客户端模型参数。
            # for name, param in c_model_global.items():
            #     param.data = param.data.to(self.device)
            for name in c_model_global:
                c_model_global[name] = c_model_global[name].to(self.device)
                # 确保全局客户端模型参数在正确的设备上。
            self.c_model_local.to(self.device)
            c_model_local = self.c_model_local.state_dict()
            # 将本地客户端模型移动到正确的设备，并获取其参数状态。
            train_kwargs['c_model_global'] = c_model_global
            train_kwargs['c_model_local'] = c_model_local
            # 将全局和本地客户端模型参数添加到训练参数字典中。
        # ========================SCAFFOLD=====================#

        iteration_cnt = 0
        # 初始化迭代次数计数器。
        for epoch in range(self.args.global_epochs_per_round):
            # 对于每轮通信的每个周期，执行本地训练。
            self.construct_mix_dataloader(share_data1, share_data2, share_y, round_idx)
            # 构建混合数据加载器。
            self.trainer.train_mix_dataloader(epoch, self.local_train_mixed_dataloader, self.device, **train_kwargs)
            # 使用混合数据加载器进行训练。
            logging.info("#############train finish for {epoch}  epoch and test result on client {index} ########".format(
                    epoch=epoch, index=self.client_index))
            # 记录训练完成的日志信息。
        # ========================SCAFFOLD=====================#
        if self.args.scaffold:
            # refer to https://github.com/Xtra-Computing/NIID-Bench/blob/HEAD/experiments.py#L403-L411
            # 这行代码检查是否启用了名为 scaffold 的参数。如果启用，下面的代码块将被执行。
            c_new_para = self.c_model_local.state_dict()
            # 这行代码保存了本地客户端模型 self.c_model_local 的当前参数状态。
            c_delta_para = copy.deepcopy(self.c_model_local.state_dict())
            # 这行代码使用 copy.deepcopy 复制了本地客户端模型的参数状态，以便稍后计算参数更新的差异。
            # global_model_para = global_model.state_dict()
            global_model_para = previous_model
            # 这行代码将全局模型的参数状态设置为 previous_model，这可能是上一轮迭代中的全局模型参数。
            # net_para = net.state_dict()

            # net_para = self.trainer.get_model_params()
            net_para = self.trainer.model.state_dict()
            # 这行代码获取了当前训练器 self.trainer 中模型的参数状态。
            if self.trainer.lr_scheduler is not None:
                current_lr = self.trainer.lr_scheduler.lr
            else:
                current_lr = self.args.lr
            # 这几行代码检查是否存在学习率调度器 lr_scheduler，如果存在则使用其当前学习率，
            # 否则使用命令行参数 args.lr 中指定的学习率。
            # current_lr = self.trainer.lr_scheduler.lr
            logging.debug(f"current_lr is {current_lr}")
            # 这行代码记录当前学习率的值。
            for key in net_para:
                # 这行代码开始一个循环，遍历网络参数字典 net_para 的所有键。
                # logging.debug(f"c_new_para[key].device : {c_new_para[key].device}, \
                #     global_model_para[key].device : {global_model_para[key].device}, \
                #     net_para[key].device : {net_para[key].device}")
                c_new_para[key] = c_new_para[key] - c_model_global[key] + \
                    (global_model_para[key].to(self.device) - net_para[key]) / (iteration_cnt * current_lr)
                # 这行代码更新了客户端模型的新参数。它通过减去全局模型和客户端模型之间的差异，
                # 然后加上全局模型和当前网络参数之间的差异，再除以迭代次数和当前学习率的乘积，来计算新的参数值。
                c_delta_para[key] = (c_new_para[key] - c_model_local[key]).to('cpu')
                # 这行代码计算了客户端模型参数的更新量，并将更新后的参数转换到CPU设备。
            self.c_model_local.load_state_dict(c_new_para)
            # 这行代码将更新后的参数加载到本地客户端模型中。
            self.trainer.model.to('cpu')
            self.c_model_local.to('cpu')
            # 这两行代码将训练器中的模型和客户端本地模型都转移到CPU设备上。
            client_other_params["c_delta_para"] = c_delta_para
            # 这行代码将计算出的参数更新量 c_delta_para 添加到客户端的其他参数字典中。
        # ========================SCAFFOLD=====================#

        weights, model_indexes = self.get_model_params()
        # 这行代码调用 get_model_params 方法来获取模型的权重和索引。
        return weights, model_indexes, self.test_data_num, client_other_params, shared_params_for_simulation  # 用于train的数据量
        # 最后一行代码返回了多个值：模型权重、模型索引、测试数据的数量、客户端的其他参数以及用于模拟的共享参数。

    def algorithm_on_train(self, share_data1, share_data2, share_y, round_idx,
            named_params, params_type='model',
            global_other_params=None,
            shared_params_for_simulation=None):
        # 这是函数的定义行，其中：
        # self 是类的实例引用。
        # share_data1 和 share_data2 可能是用于训练的数据集的一部分。
        # share_y 可能是对应的标签或输出。
        # round_idx 表示当前的训练轮次或迭代次数。
        # named_params 是一个参数字典，包含模型参数。
        # params_type 是一个字符串，默认为 'model'，用于指定参数的类型。
        # global_other_params 是一个可选参数，可能包含全局的其他参数。
        # shared_params_for_simulation 是一个可选参数，可能用于模拟的共享参数。
        if params_type == 'model':
            self.set_model_params(named_params)
            # 如果参数类型是 'model'，这行代码调用 set_model_params 方法，将 named_params 设置为模型参数。
        model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = self.fedavg_train(
                share_data1, share_data2, share_y,
                round_idx,
                global_other_params,
                shared_params_for_simulation)
        # 这行代码调用 fedavg_train 方法，传入训练数据、标签、轮次索引、全局其他参数和共享模拟参数。
        # 这个方法执行联邦平均（Federated Averaging）算法的训练过程，并返回以下内容：
        # model_params：更新后的模型参数。
        # model_indexes：模型参数的索引。
        # local_sample_number：本地样本的数量。
        # client_other_params：客户端的其他参数。
        # shared_params_for_simulation：用于模拟的共享参数。
        return model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation
        # 函数的最后一行返回从 fedavg_train 方法得到的参数和数据。
















