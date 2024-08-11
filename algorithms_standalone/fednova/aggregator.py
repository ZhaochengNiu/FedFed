from algorithms_standalone.basePS.aggregator import Aggregator
# 这行代码从 algorithms_standalone.basePS.aggregator 模块导入了 Aggregator 类。
# 这表明 Aggregator 是一个基类，可能定义了一些联邦学习中聚合器的通用行为和属性。


class FedNovaAggregator(Aggregator):
    # 定义了一个新的类 FedNovaAggregator，它继承自 Aggregator 类。
    # 这意味着 FedNovaAggregator 将继承 Aggregator 的所有方法和属性，并可以添加或重写自己的方法。
    def __init__(self,train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model):
        # 这是 FedNovaAggregator 类的构造函数，它接收以下参数：
        # self：类的实例引用。
        # train_dataloader 和 test_dataloader：分别用于训练和测试的数据加载器。
        # train_data_num 和 test_data_num：分别表示训练数据和测试数据的总数。
        # train_data_local_num_dict：一个字典，存储了每个客户端本地训练数据的数量。
        # worker_num：参与联邦学习的客户端数量。
        # device：模型运行的设备，可能是 CPU 或 GPU。
        # args：包含训练参数的配置对象。
        # model_trainer：用于模型训练的组件或对象。
        # vae_model：可选的变分自编码器模型，可能用于数据的预处理或特征提取。
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                        train_data_local_num_dict, worker_num, device,args, model_trainer, vae_model)
        # 这行代码调用了父类 Aggregator 的构造函数，并将所有传入的参数传递给它。这用于初始化从基类继承的属性。







