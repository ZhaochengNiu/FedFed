
# from .config import CfgNode as CN
from .config import CfgNode as CN

_C = CN()
# 创建一个配置节点实例。
_C.dataset = 'cifar10'
# 设置数据集为 'cifar10'。
_C.client_num_in_total = 10
# 设置总共有 10 个客户端。
_C.client_num_per_round = 5
_C.gpu_index = 0 # for centralized training or standalone usage
# 设置使用的 GPU 索引为 0。
_C.num_classes = 10
# 设置类别总数为 10。
_C.data_dir = './../data'
# 设置数据存储目录为相对路径 '../data'。
_C.partition_method = 'hetero'
# 设置数据分配方法为异构（hetero）。
_C.partition_alpha = 0.1
# 设置数据分配的 alpha 参数为 0.1。
_C.model = 'resnet18_v2'
# 设置模型为 'resnet18_v2'。
_C.model_input_channels = 3
# 设置模型输入通道数为 3。
_C.model_output_dim = 10
# 设置模型输出维度为 10。
_C.algorithm = 'FedAvg'
# 设置使用的算法为 'FedAvg'。
# fedprox
_C.fedprox = False
_C.fedprox_mu = 0.1
_C.scaffold = False

_C.global_epochs_per_round = 1
# 设置每轮通信的全局周期数为 1。
_C.comm_round = 1000
# 设置通信轮次总数为 1000。
_C.lr = 0.01
# 设置学习率为 0.01。
_C.seed = 0
# 设置随机种子为 0。
_C.record_tool = 'wandb'  # using wandb or tensorboard
_C.wandb_record = False

_C.batch_size = 64
# 设置批量大小为 64。
_C.VAE_batch_size = 64
_C.VAE_aug_batch_size = 64

_C.VAE_re = 5.0
_C.VAE_ce = 2.0
_C.VAE_kl = 0.005

_C.VAE_std1 = 0.15
_C.VAE_std2 = 0.25
# 设置变分自编码器（VAE）的两种噪声标准差。
_C.VAE_x_ce = 0.4

_C.VAE_comm_round = 15
# 设置 VAE 通信轮次为 15。
_C.VAE_client_num_per_round = 10
_C.VAE_adaptive = True
# 设置 VAE 为自适应模式。
_C.noise_type = 'Gaussian'  # Gaussian or Laplace
# 设置噪声类型为 'Gaussian'。

# ---------------------------------------------------------------------------- #
# mode settings
# ---------------------------------------------------------------------------- #
_C.mode = 'standalone'  # standalone or centralized
# 设置模式为独立（standalone）。
_C.test = True
# 设置为测试模式。
_C.instantiate_all = True
_C.client_index = 0
# 设置客户端索引为 0。
# ---------------------------------------------------------------------------- #
# task settings
# ---------------------------------------------------------------------------- #
_C.task = 'classification' #    ["classification", "stackoverflow_lr", "ptb"]
# 设置任务类型为分类（classification）。

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #

_C.dataset_aug = "default"
# 设置数据增强方法为默认。
_C.dataset_resize = False
_C.dataset_load_image_size = 32

_C.data_efficient_load = True    #  Efficiently load dataset, only load one full dataset, but split to many small ones.

_C.dirichlet_min_p = None #  0.001    set dirichlet min value for letting each client has samples of each label
_C.dirichlet_balance = False # This will try to balance dataset partition among all clients to make them have similar data amount

_C.data_load_num_workers = 1

# ---------------------------------------------------------------------------- #
# data sampler
# ---------------------------------------------------------------------------- #
_C.data_sampler = "random"  # 'random'
# 设置数据采样器为随机（random）。
_C.TwoCropTransform = False


# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #

_C.model_out_feature = False
# 设置不输出模型特征。
_C.model_out_feature_layer = "last"
_C.model_feature_dim = 512

_C.pretrained = False
# 设置不使用预训练模型。
_C.pretrained_dir = " "



# ---------------------------------------------------------------------------- #
# generator
# ---------------------------------------------------------------------------- #
_C.image_resolution = 32
# 设置图像分辨率为 32。
# ---------------------------------------------------------------------------- #
# Client Select
# ---------------------------------------------------------------------------- #
_C.client_select = "random"  #   ood_score, ood_score_oracle
# 设置客户端选择方法为随机。

# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'
# 设置损失函数为交叉熵（CrossEntropy）。
_C.exchange_model = True
# 设置为交换模型参数。


# ---------------------------------------------------------------------------- #
# optimizer settings
# comm_round is only used in FedAvg.
# ---------------------------------------------------------------------------- #
_C.max_epochs = 90

_C.client_optimizer = 'no' # Please indicate which optimizer is used, if no, set it as 'no'
_C.server_optimizer = 'no'
# 设置客户端和服务器优化器为 'no'，表示不使用特定的优化器。

_C.wd = 0.0001
_C.momentum = 0.9
# 设置权重衰减（weight decay）为 0.0001，动量为 0.9。
_C.nesterov = False


# ---------------------------------------------------------------------------- #
# Learning rate schedule parameters
# ---------------------------------------------------------------------------- #
_C.sched = 'no'   # no (no scheudler), StepLR MultiStepLR  CosineAnnealingLR
# 设置学习率调度器为 'no'，表示不使用学习率调度器。
_C.lr_decay_rate = 0.992
_C.step_size = 1
_C.lr_milestones = [30, 60]
_C.lr_T_max = 10
_C.lr_eta_min = 0
_C.lr_warmup_type = 'constant' # constant, gradual.
_C.warmup_epochs = 0
_C.lr_warmup_value = 0.1


# ---------------------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------------------- #
_C.level = 'INFO' # 'INFO' or 'DEBUG'
# 设置日志记录级别为 'INFO'。

# ---------------------------------------------------------------------------- #
# VAE settings
# ---------------------------------------------------------------------------- #
_C.VAE = True
# 设置使用变分自编码器（VAE）。
_C.VAE_local_epoch = 1
_C.VAE_d = 32
_C.VAE_z = 2048
# 设置 VAE 的本地周期数为 1，维度 d 为 32，维度 z 为 2048。
_C.VAE_sched = 'cosine'
_C.VAE_sched_lr_ate_min = 2.e-3
_C.VAE_step = '+'

_C.VAE_mixupdata = False
_C.VAE_curriculum = True # Curriculum for reconstruction term which helps for better convergence

_C.VAE_mean = 0

_C.VAE_alpha = 2.0
_C.VAE_curriculum = True
