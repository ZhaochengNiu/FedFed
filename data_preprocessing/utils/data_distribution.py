import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
# 导入了NumPy、matplotlib.pyplot、sys、os、argparse等模块。
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# 将当前目录的上上级目录添加到系统路径。
from data_preprocessing.cityscapes.data_loader import partition_data as partition_cityscapes
from data_preprocessing.pascal_voc_augmented.data_loader import partition_data as partition_pascal
# 从指定的数据预处理模块导入partition_data函数。

# 这段代码是一个Python脚本，用于可视化不同数据集划分方法下客户端的数据类别分布。以下是对代码中每个部分的详细解释：
# 整体来看，这段代码提供了一个可视化工具，用于展示不同客户端在非IID数据划分下的数据类别分布情况。
# 通过命令行参数，用户可以指定数据集、数据目录和α值，脚本将生成相应的堆叠条形图并显示和保存。

if __name__ == "__main__":
    # 定义了程序的主入口。
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5, help='Partition Alpha')
    parser.add_argument('--data_dir', type=str, default='/home/chaoyanghe/BruteForce/FedML/data/pascal_voc'
                                                        '/benchmark_RELEASE', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='pascal_voc', help="Name of dataset")

    args = parser.parse_args()
    # 使用argparse库解析命令行参数，获取α值、数据目录、数据集名称等。
    # alpha = 100
    # data_dir = "/content/data/benchmark_RELEASE/"
    # 根据数据集名称调用对应的数据划分函数。
    if args.dataset == 'pascal_voc':
        net_data_idx_map, train_data_cls_counts = partition_pascal(args.data_dir, "hetero", 4, args.alpha, 513)
    elif args.dataset == 'cityscapes':
        net_data_idx_map, train_data_cls_counts = partition_cityscapes(args.data_dir, "hetero", 4, args.alpha, 513)
    else:
        raise NotImplementedError
    # 从划分结果中提取客户端的数据，并计算每个客户端的类别分布比例。
    clients = train_data_cls_counts[0].keys()
    client1 = np.array(list(train_data_cls_counts[0].values()))
    client2 = np.array(list(train_data_cls_counts[1].values()))
    client3 = np.array(list(train_data_cls_counts[2].values()))
    client4 = np.array(list(train_data_cls_counts[3].values()))
    # Add more clients if necessary

    total = client1 + client2 + client3 + client4

    proportion_client1 = np.true_divide(client1, total) * 100
    proportion_client2 = np.true_divide(client2, total) * 100
    proportion_client3 = np.true_divide(client3, total) * 100
    proportion_client4 = np.true_divide(client4, total) * 100
    ind = [x for x, _ in enumerate(clients)]

    # 使用matplotlib的bar函数绘制堆叠条形图，展示不同客户端的类别分布。
    plt.bar(ind, proportion_client4, width=0.8, label='c4', color='b',
            bottom=proportion_client1 + proportion_client2 + proportion_client3)
    plt.bar(ind, proportion_client3, width=0.8, label='c3', color='g',
            bottom=proportion_client1 + proportion_client2)
    plt.bar(ind, proportion_client2, width=0.8, label='c2', color='silver', bottom=proportion_client1)
    plt.bar(ind, proportion_client1, width=0.8, label='c1', color='gold')

    # 设置图表的标题、X轴标签、Y轴标签、X轴刻度等。
    plt.xticks(ind, clients)
    plt.ylabel("Data")
    plt.xlabel("Classes")
    plt.title("Class Distribution: clients=4, alpha={}".format(args.alpha))
    plt.ylim = 1.0

    # rotate axis labels
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()
    plt.savefig(args.dataset + '.png')
    # 显示图表，并将其保存为PNG文件。
