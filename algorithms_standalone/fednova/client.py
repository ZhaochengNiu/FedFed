import logging
import copy

import torch

from algorithms_standalone.basePS.client import Client
# 导入所需的模块，包括日志记录模块 logging，深拷贝函数 copy，PyTorch 库 torch，以及基类 Client。


class FedNovaClient(Client):
    # 定义了 FedNovaClient 类，继承自 Client 类。
    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num, perf_timer=None, metrics=None):
        # 构造函数，初始化 FedNovaClient 类的实例。
        # 它接收多个参数，包括客户端索引、数据加载器、数据数量、设备信息、训练参数等，并调用父类构造函数。
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                test_data_num,  train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num, perf_timer, metrics)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations
        # 初始化一个字典来存储当前客户端的本地迭代次数。
        self.global_epochs_per_round = self.args.global_epochs_per_round
        # 设置全局每轮的周期数。
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round
        # 初始化一个字典来存储每个客户端每轮通信的本地周期数。

    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        # 定义学习率调度方法，根据迭代次数和预热周期调整学习率。
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

    def fednova_train(self, share_data1, share_data2, share_y,round_idx=None, shared_params_for_simulation=None):
        # 定义 FedNova 算法的训练方法。这个方法接收共享数据和其他参数，执行训练并计算更新。
        previous_model = copy.deepcopy(self.trainer.get_model_params())
        # 在训练开始前，深拷贝当前模型参数，以便后续计算。
        client_other_params = {}
        self.move_to_gpu(self.device)

        tau = 0
        for epoch in range(self.args.global_epochs_per_round):
            # 初始化 tau 并进行每轮的训练周期。
            self.construct_mix_dataloader(share_data1, share_data2, share_y,round_idx)
            # 构建混合数据加载器，用于训练。
            self.trainer.train_mix_dataloader(epoch, self.local_train_mixed_dataloader, self.device)
            tau = len(self.local_train_mixed_dataloader)
            # 设置 tau 为混合数据加载器的长度。
            logging.info("#############train finish for {epoch}  epoch and test result on client {index} ########".format(
                    epoch=epoch, index=self.client_index))

        a_i = (tau - self.args.momentum * (1 - pow(self.args.momentum, tau)) / (1 - self.args.momentum)) / (1 - self.args.momentum)
        # 根据 tau 和动量参数 momentum 计算 a_i。
        global_model_para = previous_model
        net_para = self.trainer.get_model_params()
        # 获取全局模型参数和当前网络参数。
        norm_grad = copy.deepcopy(previous_model)
        for key in norm_grad:
            norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
            # 计算归一化梯度。

        self.move_to_cpu()
        # 将模型参数移动到 CPU。
        client_other_params["a_i"] = a_i
        client_other_params["norm_grad"] = norm_grad
        # 将 a_i 和归一化梯度添加到客户端其他参数中。
        # return None, None, self.local_sample_number, a_i, norm_grad
        return None, None, self.local_sample_number, client_other_params, shared_params_for_simulation
        # 返回训练结果，包括本地样本数量和其他客户端参数。

    def algorithm_on_train(self, update_state_kargs,
            client_index, named_params, params_type='model', traininig_start=False,
            shared_params_for_simulation=None):
        # 定义 algorithm_on_train 方法，但当前实现为空。这个方法可能用于在训练过程中更新客户端状态。
        pass
















