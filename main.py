import argparse
# 导入Python的argparse模块，用于解析命令行参数。
import logging
# 导入Python的logging模块，用于记录日志。
import os
# 导入Python的os模块，用于操作系统功能，如文件路径。
import socket
# 导入Python的socket模块，用于获取主机名。
import sys
# 导入Python的sys模块，用于访问与Python解释器相关的变量和函数。

import numpy as np
# 导入numpy库并命名为np，numpy是一个用于科学计算的库。
import torch
# 导入PyTorch库，用于深度学习。
# add the FedML root directory to the python path

from utils.logger import logging_config
# 从 utils/logger.py 模块导入 logging_config 函数。
from configs import get_cfg
# 从 configs 模块导入 get_cfg 函数，用于获取配置。
from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
# 从 algorithms_standalone/fedavg/FedAVGManager.py 导入 FedAVGManager 类，用于联邦平均算法的管理。
from algorithms_standalone.fednova.FedNovaManager import FedNovaManager
# 从 algorithms_standalone/fednova/FedNovaManager.py 导入 FedNovaManager 类，用于联邦Nova算法的管理。
from utils.set import *
# 从utils/set.py模块导入所有内容。
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


def add_args(parser):
    # 定义一个函数add_args，用于向argparse.ArgumentParser对象添加所需的参数。
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # 函数 add_args 的实现，它添加了 --config_file 和 opts 两个参数，并解析命令行参数。
    return args


if __name__ == "__main__":
    # 这是Python的常规用法，表示如果这个脚本是作为主程序运行的，下面的代码将被执行。
    # initialize distributed computing (MPI)
    # parse python script input parameters

    #----------loading personalized params-----------------#
    parser = argparse.ArgumentParser()
    # 创建一个ArgumentParser对象。
    args = add_args(parser)
    # 调用add_args函数，添加参数并解析。
    print(args.config_file)
    # 打印出配置文件的路径。
    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    cfg.setup(args)
    # 设置配置，首先获取默认配置，然后根据命令行参数和配置文件设置配置。
    # Build config once again
    #cfg.setup(args)
    cfg.mode = 'standalone'
    # 设置配置模式为独立模式。
    cfg.server_index = -1
    cfg.client_index = -1
    seed = cfg.seed
    process_id = 0
    # 设置服务器和客户端的索引，以及进程ID。
    # show ultimate config
    logging.info(dict(cfg))
    # 打印最终的配置信息。
    #-------------------customize the process name-------------------
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)
    logging_config(args=cfg, process_id=process_id)
    # 自定义进程名称并设置日志配置。
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()))
    # 获取主机名并记录日志。
    set_random_seed(seed)
    # 设置随机种子以确保结果的可复现性。
    torch.backends.cudnn.deterministic =True
    # 确保PyTorch的cudnn后端是确定性的。
    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    # 设置设备，优先使用GPU。
    if cfg.record_tool == 'wandb' and cfg.wandb_record:
        import wandb
        wandb.init(config=args, name='test',
                   project='CIFAR10')
    else: 
        os.environ['WANDB_MODE'] = 'dryrun'
        # 如果配置指定使用wandb记录实验，初始化wandb；否则设置环境变量以禁用wandb。
    if cfg.algorithm == 'FedAvg':
        fedavg_manager = FedAVGManager(device, cfg)
        fedavg_manager.train()
    elif cfg.algorithm == 'FedNova':
        fednova_manager = FedNovaManager(device, cfg)
        fednova_manager.train()
    else:
        raise NotImplementedError
        # 据配置的算法类型，实例化相应的管理器并调用其train方法开始训练。
        # 如果算法类型不是FedAvg或FedNova，则抛出一个NotImplementedError异常。







