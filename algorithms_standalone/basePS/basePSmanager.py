import logging
from abc import  abstractmethod
# 导入 Python 的日志模块和抽象基类模块。
import numpy as np
import torch
# 导入 NumPy 库和 PyTorch 库。
from model.build import create_model
# 从模型构建模块导入 create_model 函数。
from utils.data_utils import (
    get_selected_clients_label_distribution,
    average_named_params
)
# 从数据实用工具模块导入 get_selected_clients_label_distribution 和 average_named_params 函数。
from data_preprocessing.build import load_data
# 从数据预处理构建模块导入 load_data 函数。
from data_preprocessing.cifar10.datasets import Dataset_Personalize_4Tensor
# 从 CIFAR10 数据集模块导入 Dataset_Personalize_4Tensor 类。
from utils.tool import *
from model.build import create_model
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
# 导入工具函数、CIFAR10 数据集类和变换模块。


class BasePSManager(object):
    # 定义了 BasePSManager 抽象基类。
    def __init__(self, device, args):
        # 构造函数，初始化设备和参数。
        self.device = device
        self.args = args
        # ================================================
        self._setup_datasets()
        # 调用 _setup_datasets 方法来设置数据集。
        self.selected_clients = None
        # 初始化选中客户端列表。
        self.client_list = []
        # 初始化客户端列表。
        self.aggregator = None
        # 初始化聚合器，将在服务器设置中配置。
        # ================================================
        if self.args.instantiate_all:   
            self.number_instantiated_client = self.args.client_num_in_total
            # 根据参数设置实例化客户端的数量。
        else:
            self.number_instantiated_client = self.args.client_num_per_round
        self._setup_clients()
        # 调用 _setup_clients 方法来设置客户端。
        # ================================================
        self._setup_server()
        # 调用 _setup_server 方法来设置服务器。
        # aggregator will be initianized in _setup_server()
        self.comm_round = self.args.comm_round
        # 设置通信轮次。
        # ================================================
        #    logging all acc
        self.test_acc_list = []
        # 初始化测试准确率列表。
        self._share_data_step()
        # 定义 _setup_datasets 方法来加载和设置数据集。

# got it
    def _setup_datasets(self):
        # dataset = load_data(self.args, self.args.dataset)

        train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl,train_data_local_num_dict, \
        test_data_local_num_dict, test_data_local_dl_dict, train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
        other_params = load_data(load_as="training", args=self.args, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                                 dirichlet_balance=False, dirichlet_min_p=None,dataset=self.args.dataset, datadir=self.args.data_dir,
                                 partition_method=self.args.partition_method, partition_alpha=self.args.partition_alpha,
                                 client_number=self.args.client_num_in_total, batch_size=self.args.batch_size, num_workers=self.args.data_load_num_workers,
                                 data_sampler=self.args.data_sampler,resize=self.args.dataset_load_image_size, augmentation=self.args.dataset_aug)
        # 调用 load_data 函数来加载训练和测试数据。

        self.other_params = other_params
        self.train_data_global_dl = train_data_global_dl
        self.test_data_global_dl = test_data_global_dl

        self.train_data_global_num = train_data_global_num
        self.test_data_global_num = test_data_global_num

        self.test_data_local_dl_dict = test_data_local_dl_dict   # {client_idx: client_idx_test_dataloader}
        self.train_data_local_num_dict = train_data_local_num_dict  # {client_idx: client_idx_train_num}
        self.test_data_local_num_dict = test_data_local_num_dict # {client_idx: client_idx_test_num}
        self.train_data_local_ori_dict = train_data_local_ori_dict  # {client_idx: client_idx_train_ori_data}
        self.train_targets_local_ori_dict = train_targets_local_ori_dict # {client_idx: client_idx_ori_targets}
        self.client_dataidx_map = other_params['client_dataidx_map']
        self.train_cls_local_counts_dict = other_params['train_cls_local_counts_dict']

        self.class_num = class_num

        if "train_cls_local_counts_dict" in self.other_params:
            # 字典嵌套字典
            self.train_cls_local_counts_dict = self.other_params["train_cls_local_counts_dict"]  # {client_idx:{label0: labe0_num_client_idx,...,label9: labe9_num_client_idx}}
            # Adding missing classes to list
            classes = list(range(self.class_num)) # [0,1,2,3,4,...,class_num - 1]
            for key in self.train_cls_local_counts_dict:
                # key means the client index
                if len(classes) != len(self.train_cls_local_counts_dict[key]):
                    # print(len(classes))
                    # print(len(train_data_cls_counts[key]))
                    add_classes = set(classes) - set(self.train_cls_local_counts_dict[key])
                    # print(add_classes)
                    for e in add_classes:
                        self.train_cls_local_counts_dict[key][e] = 0   
        else:
            self.train_cls_local_counts_dict = None

    def _setup_server(self):
        # 定义 _setup_server 方法，用于设置服务器端，当前实现为空。
        pass

    def _setup_clients(self):
        # 定义 _setup_clients 方法，用于设置客户端，当前实现为空。
        pass

    def _share_data_step(self):
        # 定义 _share_data_step 方法，用于在 VAE 步骤中共享数据。
        for round in range(self.args.VAE_comm_round):
        # 遍历 VAE 通信轮次。
# -------------------train VAE for every client----------------#
            logging.info("############Round {} VAE #######################".format(round))

# ----------------- sample client duiring VAE step------------------#
            client_indexes = self.client_sample_for_VAE(round, self.args.client_num_in_total, self.args.VAE_client_num_per_round)
            # 调用 client_sample_for_VAE 方法来抽样客户端。
            for client_index in client_indexes:
                client = self.client_list[client_index]
                client.train_vae_model(round)

#------------------aggregate VAE from sampled client----------------------------------------#
            self._aggregate_sampled_client_vae(client_indexes, round)  # using
            # 调用 _aggregate_sampled_client_vae 方法来聚合客户端 VAE。

            self.aggregator.test_on_server_by_vae(round)
        # vae_model = torch.load("vae_model_client100_alpha0.1_datasetcifar100.pth")
        for client in self.client_list:
            client.generate_data_by_vae()
          
        for client in self.client_list:
            del client.vae_model
        self._get_local_shared_data()

        self.aggregator.save_vae_param()

    def _aggregate_sampled_client_vae(self,client_indexes, round):
        # 定义 _aggregate_sampled_client_vae 方法来聚合客户端 VAE。
        model_list = []
        training_data_num = 0
        data_num_list = []
        aggregate_weight_list = []
        for client_index in client_indexes:
            client = self.client_list[client_index]
            model_list.append((client.local_sample_number, client.get_vae_para()))
            data_num_list.append(client.local_sample_number)
            training_data_num += client.local_sample_number
        for i in range(0, len(data_num_list)):
            local_sample_number = data_num_list[i]
            weight_by_sample_num = local_sample_number / training_data_num
            aggregate_weight_list.append(weight_by_sample_num)

        averaged_vae_params = average_named_params(
            model_list,  # from sampled client model_list  [(sample_number, model_params)]
            aggregate_weight_list
        )
        self.aggregator.set_vae_param(averaged_vae_params)
        logging.info("initial global model using Classifier from VAE in Round {}".format(round))

#    distribute VAE model to all clients
        for client in self.client_list:
            client.set_vae_para(averaged_vae_params)

    def _get_local_shared_data(self):
        # 定义 _get_local_shared_data 方法来获取本地共享数据。
        # in test step using two types shared data
        for client_idx in range(len(self.client_list)):
            client_data1, data_y = self.client_list[client_idx].get_local_share_data(noise_mode=1)
            client_data2, _ = self.client_list[client_idx].get_local_share_data(noise_mode=2)

            if client_idx == 0:
                self.global_share_dataset1 = client_data1
                self.global_share_dataset2 = client_data2
                self.global_share_data_y = data_y
            else:
                self.global_share_dataset1 = torch.cat((self.global_share_dataset1, client_data1))
                self.global_share_dataset2 = torch.cat((self.global_share_dataset2, client_data2))
                self.global_share_data_y = torch.cat((self.global_share_data_y, data_y))

    def test(self):
        # 定义 test 方法来进行测试。
        logging.info("################test_on_server_for_all_clients : {}".format(
            self.server_timer.global_outer_epoch_idx))
        avg_acc = self.aggregator.test_on_server_for_all_clients(
            self.server_timer.global_outer_epoch_idx, self.total_test_tracker, self.metrics)

        return avg_acc

    def get_init_state_kargs(self):
        # 定义 get_init_state_kargs 方法来获取初始化状态参数。
        self.selected_clients = [i for i in range(self.args.client_num_in_total)]
        init_state_kargs = {}
        return init_state_kargs

    def get_update_state_kargs(self):
        # 定义 get_update_state_kargs 方法来获取更新状态参数。
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            update_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            update_state_kargs = {}
        return update_state_kargs

 # ----------------- sample clinet duiring VAE step------------------#
    def client_sample_for_VAE(self, round_idx, client_num_in_total, client_num_per_round):
        # 定义 client_sample_for_VAE 方法来抽样 VAE 客户端。
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            np.random.seed(self.args.VAE_comm_round - round_idx)
            if self.args.client_select == "random":
                num_clients = min(client_num_per_round, client_num_in_total)
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        logging.info("VAE sampling client_indexes = %s" % str(client_indexes))
        return client_indexes

    def lr_schedule(self, num_iterations, warmup_epochs):
        # 定义 lr_schedule 方法来设置学习率调度。
        epochs = self.server_timer.global_outer_epoch_idx
        iterations = self.server_timer.global_outer_iter_idx

        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.aggregator.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.aggregator.trainer.lr_schedule(epochs)


    # ==============train clients and add results to aggregator ===s================================
    def train(self):
        # 定义 train 方法来进行训练。
        for round in range(self.comm_round):

            logging.info("################Communication round : {}".format(round))
            # w_locals = []

            # Note in the first round, something of global_other_params is not constructed by algorithm_train(),
            # So care about this.
            if round == 0:
                named_params = self.aggregator.get_global_model_params()
                params_type = 'model'
                global_other_params = {}
                shared_params_for_simulation = {}


                # ========================SCAFFOLD=====================#
                if self.args.scaffold:
                    c_global_para = self.aggregator.c_model_global.state_dict()
                    global_other_params["c_model_global"] = c_global_para
                # ========================SCAFFOLD=====================#

# ----------------- sample clinet saving in manager------------------#
            client_indexes = self.aggregator.client_sampling(   
                round, self.args.client_num_in_total,
                self.args.client_num_per_round)

            update_state_kargs = self.get_update_state_kargs()   


# -----------------train model using algorithm_train and aggregate------------------#
            named_params, params_type, global_other_params, shared_params_for_simulation = self.algorithm_train(
                round,
                client_indexes,
                named_params,
                params_type,
                global_other_params,
                update_state_kargs,
                shared_params_for_simulation
            )
# -----------------test model on server every communication round------------------#
            avg_acc = self.aggregator.test_on_server_for_round(self.args.VAE_comm_round+round)
            self.test_acc_list.append(avg_acc)
            print(avg_acc)
            if round % 20 == 0:
                print(self.test_acc_list)
        
        self.aggregator.save_classifier()



    @abstractmethod
    def algorithm_train(self, round_idx, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, 
                        shared_params_for_simulation):
        # 定义 algorithm_train 抽象方法，用于训练算法。
        pass


    # ===========================================================================















