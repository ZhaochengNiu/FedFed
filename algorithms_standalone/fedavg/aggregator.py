from algorithms_standalone.basePS.aggregator import Aggregator
from model.build import create_model
# 这两行代码导入了所需的模块。
# Aggregator 类可能定义了服务器端聚合器的基本行为，而 create_model 函数用于创建模型实例。


class FedAVGAggregator(Aggregator):
    # 定义了一个名为 FedAVGAggregator 的类，它继承自 Aggregator 类。
    def __init__(self,train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model):
        # 这是 FedAVGAggregator 类的构造函数，它接收多个参数来初始化聚合器的属性。
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                        train_data_local_num_dict, worker_num, device,args, model_trainer, vae_model)
        # 调用父类 Aggregator 的构造函数，传递所有必要的参数。
        if self.args.scaffold:
            # 检查是否启用了 scaffold 参数。
            self.c_model_global = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            # 如果启用了 scaffold，则使用 create_model 函数创建一个全局客户端模型。
            for name, params in self.c_model_global.named_parameters():
                params.data = params.data*0
                # 将新创建的全局客户端模型的所有参数置零。

    def get_max_comm_round(self):
        # 定义了一个名为 get_max_comm_round 的方法，用于获取最大通信轮次。
        return self.args.comm_round
        # 返回 args 中定义的最大通信轮次 comm_round。
