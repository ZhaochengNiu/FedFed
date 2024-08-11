import copy
import logging
# 导入 Python 标准库中的 copy 和 logging 模块，分别用于复制对象和日志记录。
from .client import FedAVGClient
# 从当前包中的client模块导入FedAVGClient类。
from .aggregator import FedAVGAggregator
# 从当前包中的aggregator模块导入FedAVGAggregator类。
from utils.data_utils import (
    get_avg_num_iterations,
)
# 从 utils 包的 data_utils 模块导入一些数据处理相关的工具函数。
from algorithms_standalone.basePS.basePSmanager import BasePSManager
# 从 algorithms_standalone 包的 basePS 模块导入 BasePSManager 类。
from model.build import create_model
# 从 model 包的 build 模块导入 create_model 函数，用于创建模型。
from trainers.build import create_trainer
# 从 trainers 包的 build 模块导入 create_trainer 函数，用于创建训练器。
from model.FL_VAE import *
# 从 model 包的 FL_VAE 模块导入所有内容，这可能包含变分自编码器（VAE）相关的类和函数。


class FedAVGManager(BasePSManager):
    # 定义了一个名为 FedAVGManager 的类，它继承自 BasePSManager。
    def __init__(self, device, args):
        super().__init__(device, args)
        # FedAVGManager 类的构造函数，初始化设备和参数。
        self.global_epochs_per_round = self.args.global_epochs_per_round
        # 设置每轮全局训练的周期数。

    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        # 定义了私有方法_setup_server，用于设置服务器，并记录日志。
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, **self.other_params)
        # 调用create_model函数创建模型。
        init_state_kargs = {} 
        if self.args.VAE:
            VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)
            # 如果参数中指定使用 VAE，创建一个 FL_CVAE_cifar 模型实例。
        model_trainer = create_trainer(   
            self.args, self.device, model,train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num,server_index=0, role='server',**init_state_kargs
        )
        # 调用create_trainer函数创建服务器端的训练器。
        self.aggregator = FedAVGAggregator(self.train_data_global_dl, self.test_data_global_dl, self.train_data_global_num,
                self.test_data_global_num, self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer,VAE_model)
        # 创建FedAVGAggregator聚合器实例。
        logging.info("############_setup_server (END)#############")
        # 记录日志，表示服务器设置结束。

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        # 定义了私有方法_setup_clients，用于设置客户端，并记录日志。
        init_state_kargs = self.get_init_state_kargs()
        # 获取客户端初始化状态的参数。
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            # 遍历客户端索引，创建每个客户端。
            if self.args.VAE:
                VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)
                # 如果使用VAE，为每个客户端创建VAE模型实例。
            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, **self.other_params)
            # 为每个客户端创建模型。
            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)   
            # 计算每个客户端的迭代次数。
            model_trainer = create_trainer(self.args, self.device, model,class_num=self.class_num,
                                           other_params=self.other_params,client_index=client_index, role='client',
                                           **init_state_kargs)
            # 为每个客户端创建训练器。
            client = FedAVGClient(client_index,train_ori_data=self.train_data_local_ori_dict[client_index],
                             train_ori_targets=self.train_targets_local_ori_dict[client_index],
                             test_dataloader=self.test_data_local_dl_dict[client_index],
                             train_data_num=self.train_data_local_num_dict[client_index],
                             test_data_num=self.test_data_local_num_dict[client_index],
                             train_cls_counts_dict = self.train_cls_local_counts_dict[client_index],
                             device=self.device, args=self.args, model_trainer=model_trainer,
                             vae_model=VAE_model,dataset_num=self.train_data_global_num)
            # 创建FedAVGClient客户端实例，并设置其属性。
            # client.train_vae_model()
            self.client_list.append(client)
            # 将客户端实例添加到客户端列表。
        logging.info("############setup_clients (END)#############")
        # 记录日志，表示客户端设置结束。

    # override
    def check_end_epoch(self):
        # 定义check_end_epoch方法，用于检查是否结束一个周期。
        return True

    def algorithm_train(self, round_idx, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, 
                        shared_params_for_simulation):
        # 定义 algorithm_train 方法，这是算法训练的主要逻辑。
        for i, client_index in enumerate(client_indexes):  # i is index in client_indexes, client_index is the sampled client in original client list
            copy_global_other_params = copy.deepcopy(global_other_params)
            # 遍历客户端索引，进行客户端本地训练。
            if self.args.exchange_model:    # copy model params
                copy_named_model_params = copy.deepcopy(named_params)
                # 如果启用模型参数交换，复制全局模型参数。
            if self.args.instantiate_all:
                client = self.client_list[client_index]
            else:
                client = self.client_list[i]

            if round_idx == 0:
                traininig_start = True
            else:
                traininig_start = False    # 是不是第一轮
            # 如果是第一轮训练，设置traininig_start为True。
            # client training.............
            '''
                   return:
                   @named_params:   all the parameters in model: {parameters_name: parameters_values}
                   @model_indexes:  None
                   @local_sample_number: the number of traning set in local 
                   @other_client_params: in FedAvg is {}
                   @local_train_tracker_info:
                   @local_time_info:  using this by local_time_info['local_time_info'] = {client_index:   , local_comm_round_idx:,   local_outer_epoch_idx:,   ...}
                   @shared_params_for_simulation: not using in FedAvg
                   '''
            model_params, model_indexes, local_sample_number, client_other_params, \
             shared_params_for_simulation = \
                    client.train(self.global_share_dataset1,self.global_share_dataset2, self.global_share_data_y,
                                 round_idx, copy_named_model_params, params_type,
                                 copy_global_other_params,
                                 shared_params_for_simulation=shared_params_for_simulation)

            self.aggregator.add_local_trained_result(
                client_index, model_params, model_indexes, local_sample_number, client_other_params)
            # 聚合器 add_local_trained_result 方法添加本地训练结果。
        '''
             return:
             @averaged_params:
             @global_other_params:
             @shared_params_for_simulation:
             '''
        global_model_params, global_other_params, \
        shared_params_for_simulation = self.aggregator.aggregate()
        # 聚合器 aggregate 方法聚合客户端的训练结果。
        params_type = 'model'

        #-----------------------distribute updated model----------------#
        logging.info("distribute the updated model to all clients")
        for client_index in range(len(self.client_list)):
            self.client_list[client_index].set_model_params(global_model_params)
        # 更新后的全局模型参数分发到所有客户端。
        return global_model_params, params_type, global_other_params, shared_params_for_simulation
        # 返回聚合后的全局模型参数和其他信息。





