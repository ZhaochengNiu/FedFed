import copy
# 导入 copy 模块，用于复制列表。

# 这段代码定义了一个名为 Averager 的类，它负责实现权重的平均计算，可能用于在联邦学习中对客户端模型权重进行平均。
# 以下是对代码的逐行解释：
# 整体来看，Averager 类实现了一个简单的平均权重计算方法，它根据客户端的样本数量来分配权重，并将这些权重用于模型的聚合。
# 这个方法可能在联邦学习中的模型更新聚合步骤非常有用。


class Averager(object):
    """
        Responsible to implement average.
        There maybe some history information need to be memorized.
        类文档字符串，说明 Averager 类负责实现平均值的计算，并且可能需要记忆一些历史信息。
    """
    # 定义了一个名为 Averager 的类，它继承自 object（在 Python 3 中所有类都隐式地继承自 object，所以这里的继承是可选的）。
    def __init__(self, args, model):
        self.args = args
    # 构造函数，初始化 Averager 类的实例，并接收参数 args 和 model。这里没有使用 model，但它可能在类的其他方法中被使用。

    def get_average_weight(self, sample_num_list, avg_weight_type='datanum'):
        # 定义了一个名为 get_average_weight 的方法，用于获取平均权重。它接收以下参数：
        # sample_num_list：一个列表，包含每个客户端的样本数量。
        # avg_weight_type：一个字符串，指定权重类型的平均方式，默认为 'datanum'。
        # balance_sample_number_list = []
        average_weights_dict_list = []
        sum = 0
        inv_sum = 0 
        # 初始化一个空列表 average_weights_dict_list 用于存储平均权重，以及变量 sum 和 inv_sum。
        sample_num_list = copy.deepcopy(sample_num_list)
        # 复制 sample_num_list 以避免修改原始列表。
        # for i in range(0, len(sample_num_list)):
        #     sample_num_list[i] * np.random.random(1)

        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]
            inv_sum = None
            sum += local_sample_number
        # 计算所有客户端样本数量的总和。
        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]

            if avg_weight_type == 'datanum':
                weight_by_sample_num = local_sample_number / sum
                # 根据样本数量计算每个客户端的权重。
            average_weights_dict_list.append(weight_by_sample_num)
            # 将计算得到的权重添加到 average_weights_dict_list 列表中。
        homo_weights_list = average_weights_dict_list
        # 创建一个与平均权重列表相同的列表 homo_weights_list，这里可能是为了保持一致性或后续处理。
        return average_weights_dict_list, homo_weights_list
        # 返回两个列表：average_weights_dict_list 和 homo_weights_list。












