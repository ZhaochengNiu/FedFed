import logging
import torch
import numpy as np
# 导入了日志记录库、PyTorch 库和 NumPy 库

# 这段代码定义了一个名为 ImbalancedDatasetSampler 的类，
# 它继承自 torch.utils.data.sampler.Sampler，用于处理数据集中的类别不平衡问题。以下是对代码中每个部分的详细解释：
# 整体来看，ImbalancedDatasetSampler 类通过为不同类别的样本分配不同的权重，解决了数据集中的类别不平衡问题。
# 这有助于在训练机器学习模型时，给予少数类别更多的关注。


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    # 定义了一个采样器类，用于在数据加载过程中处理不平衡数据集。
    def __init__(self, args, dataset, indices=None, num_samples=None, class_num=10, **kwargs):
        # 构造函数接受参数 args（可能包含算法参数）、dataset（数据集对象）、indices（可选，数据索引列表）、
        # num_samples（每次迭代要抽取的样本数量）、class_num（类别数量）和其他关键字参数。
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        logging.info("self.indices: {}".format(self.indices))
        # 使用日志记录记录 self.indices 和 label_to_count 的信息。
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # 如果没有提供 indices，则使用数据集中所有元素的索引；如果没有提供 num_samples，则在每次迭代中抽取 len(indices) 个样本。
        self.args = args
        self.dataset = dataset
        # distribution of classes in the dataset 
        # label_to_count = [0] * len(np.unique(dataset.target))
        label_to_count = [0] * class_num
        logging.info("label_to_count: {}".format(label_to_count))
        # 使用日志记录记录 self.indices 和 label_to_count 的信息。
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            # 初始化一个列表来存储每个类别的样本数量，然后遍历索引列表，统计每个类别的样本数
        for i in range(len(label_to_count)):
            if label_to_count[i] == 0:
                label_to_count[i] = 1
                # 确保 label_to_count 中没有元素为零，避免后续计算出错。
        self.label_to_count = label_to_count

        effective_num = 1.0 - np.power(self.beta, label_to_count)
        per_cls_weights = (1.0 - self.beta) / np.array(effective_num)
        # 根据类别的样本数量计算每个类别的权重，使用 beta 参数调整权重分布。
        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        # 为每个样本计算权重，并将权重存储在 self.weights 张量中。

    def _get_label(self, dataset, idx):
        # 根据数据集和索引获取样本的标签。
        return dataset.target[idx]

    def update(self, **kwargs):
        # 这个方法用于更新采样器的状态，当前实现为空。
        pass

    def __iter__(self):
        # 这个方法实现了迭代器协议，使用 torch.multinomial 根据权重随机抽取样本索引。
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        # 返回每次迭代中抽取的样本数量。
        return self.num_samples