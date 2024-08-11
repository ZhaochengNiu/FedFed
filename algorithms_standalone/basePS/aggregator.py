import os
import sys
# 导入 os 和 sys 模块，分别用于操作系统功能和 Python 运行时环境。
from algorithms.basePS.ps_aggregator import PSAggregator
# 从 algorithms.basePS.ps_aggregator 模块导入 PSAggregator 类。
# 这表明 PSAggregator 是一个基类，可能定义了参数服务器（Parameter Server）聚合器的通用行为。
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# 将上级目录的上级（可能是项目的根目录）添加到模块搜索路径的开头。这通常用于确保可以正确导入项目中的模块。


class Aggregator(PSAggregator):
    # 定义了一个名为 Aggregator 的新类，它继承自 PSAggregator 类。
    # 这意味着 Aggregator 将扩展或修改基类 PSAggregator 的行为。
    def __init__(self, train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model):
        # 这是 Aggregator 类的构造函数，它接收以下参数：
        # train_dataloader 和 test_dataloader：分别用于训练和测试的数据加载器。
        # train_data_num 和 test_data_num：分别表示训练数据和测试数据的总数。
        # train_data_local_num_dict：一个字典，存储了每个客户端本地训练数据的数量。
        # worker_num：参与训练的工作者或客户端的数量。
        # device：模型运行的设备，可能是 CPU 或 GPU。
        # args：包含训练参数的配置对象。
        # model_trainer：用于模型训练的组件或对象。
        # vae_model：可选的变分自编码器模型，可能用于数据的预处理或特征提取。
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model)
        # 调用父类 PSAggregator 的构造函数，使用所有传入的参数来初始化继承的属性。

    def get_max_comm_round(self):
        # 定义了一个名为 get_max_comm_round 的方法，但当前实现为空。
        # 这个方法可能用于获取最大通信轮次，这在联邦学习或分布式训练中是一个重要的参数。
        pass






















