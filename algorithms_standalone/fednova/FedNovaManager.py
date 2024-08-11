import copy
import logging

import torch

from .client import FedNovaClient
from .aggregator import FedNovaAggregator
from utils.data_utils import get_avg_num_iterations
from model.FL_VAE import *
from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer
# 导入所需的模块和类。


class FedNovaManager(BasePSManager):
    # 定义了 FedNovaManager 类，继承自 BasePSManager。
    def __init__(self, device, args):
        # 构造函数，初始化 FedNovaManager 类的实例，接收设备和参数。
        super().__init__(device, args)
        # 调用父类构造函数进行初始化。
        self.global_epochs_per_round = self.args.global_epochs_per_round
        # 设置全局每轮的周期数。

    def _setup_server(self):
        # 定义设置服务器的方法。
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                             device=self.device, **self.other_params)
        # 创建模型实例。
        init_state_kargs = {}  
        if self.args.VAE:
            VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

        model_trainer = create_trainer( 
            self.args, self.device, model, train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num, server_index=0, role='server', **init_state_kargs
        )

        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedNovaAggregator(self.train_data_global_dl, self.test_data_global_dl,
                                           self.train_data_global_num,
                                           self.test_data_global_num, self.train_data_local_num_dict,
                                           self.args.client_num_in_total, self.device,
                                           self.args, model_trainer, VAE_model)

        # 创建并初始化聚合器 FedNovaAggregator。
        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        # 定义设置客户端的方法。
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()  
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            # 遍历客户端索引，设置每个客户端。
            if self.args.VAE:
                VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                                 device=self.device, **self.other_params)

            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict,
                                                    self.args.batch_size)  
            model_trainer = create_trainer(self.args, self.device, model, class_num=self.class_num,
                                           other_params=self.other_params, client_index=client_index, role='client',
                                           **init_state_kargs)

            client = FedNovaClient(client_index, train_ori_data=self.train_data_local_ori_dict[client_index],
                                  train_ori_targets=self.train_targets_local_ori_dict[client_index],
                                  test_dataloader=self.test_data_local_dl_dict[client_index],
                                  train_data_num=self.train_data_local_num_dict[client_index],
                                  test_data_num=self.test_data_local_num_dict[client_index],
                                  train_cls_counts_dict=self.train_cls_local_counts_dict[client_index],
                                  device=self.device, args=self.args, model_trainer=model_trainer,
                                  vae_model=VAE_model, dataset_num=self.train_data_global_num,
                                  perf_timer=self.perf_timer, metrics=self.metrics)
            # client.train_vae_model()
            # 创建并初始化每个 FedNovaClient 实例。
            self.client_list.append(client)
            # 将客户端添加到客户端列表。
        logging.info("############setup_clients (END)#############")

    # override
    def check_end_epoch(self):
        # 定义检查是否结束周期的方法。
        return True

    def train(self):
        # 定义训练方法，进行联邦学习的通信轮次。
        for round in range(self.comm_round):
            # 遍历通信轮次。
            logging.debug("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
            # w_locals = []
            global_model_params = self.aggregator.get_global_model_params()
            # 获取全局模型参数。
            client_indexes = self.aggregator.client_sampling(
                round, self.args.client_num_in_total,
                self.args.client_num_per_round)
            # 通过聚合器进行客户端采样。
            logging.debug("client_indexes = " + str(client_indexes))
            a_list = {}
            d_list = {}
            n_list = {}
            # 初始化列表，用于存储客户端的 a_i、d_i 和本地样本数量 n_i。
            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            for i, client_index in enumerate(client_indexes):
                # 遍历选中的客户端索引。
                if self.args.instantiate_all:
                    client = self.client_list[client_index]
                else:
                    client = self.client_list[i]

                if self.args.exchange_model:
                    copy_global_model_params = copy.deepcopy(global_model_params)
                    client.set_model_params(copy_global_model_params) 
                client.move_to_gpu(self.device)


                global_other_params = {}
                shared_params_for_simulation = {}

                model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = \
                    client.fednova_train( self.global_share_dataset1, self.global_share_dataset2, self.global_share_data_y,
                                          round_idx=client.client_timer.global_comm_round_idx,
                                        shared_params_for_simulation=shared_params_for_simulation)
                    # 在客户端上执行 FedNova 训练。
                a_i, d_i = client_other_params["a_i"], client_other_params["norm_grad"]

                client.move_to_cpu()
                a_list[client_index] = a_i
                d_list[client_index] = d_i
                n_i = local_sample_number
                n_list[client_index] = n_i

            total_n = sum(n_list.values())
            d_total_round = copy.deepcopy(global_model_params)
            # 深拷贝全局模型参数，用于计算更新。
            for key in d_total_round:
                d_total_round[key] = 0.0
            for client_index in client_indexes:
                # 遍历客户端索引，计算全局更新。
                d_para = d_list[client_index]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[client_index] / total_n

            # update global model
            coeff = 0.0
            for client_index in client_indexes:
                coeff = coeff + a_list[client_index] * n_list[client_index] / total_n

            # global_model_params = global_model.state_dict()
            for key in global_model_params:
                #print(updated_model[key])
                if global_model_params[key].type() == 'torch.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif global_model_params[key].type() == 'torch.cuda.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    global_model_params[key] -= coeff * d_total_round[key]
            self.aggregator.set_global_model_params(global_model_params)
            # 将更新后的全局模型参数设置回聚合器。
            # -----------------test model on server every communication round------------------#
            avg_acc = self.aggregator.test_on_server_for_round(self.args.VAE_comm_round + round)
            # 在服务器上测试模型的平均准确率。
            self.test_acc_list.append(avg_acc)
            print(avg_acc)
            if round % 20 == 0:
                print(self.test_acc_list)

    def algorithm_train(self, client_indexes, named_params, params_type,
                        update_state_kargs, global_time_info, shared_params_for_simulation):
        # 定义算法训练的方法，但当前实现为空。
        pass




