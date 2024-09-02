import os
import logging
from .loader import Data_Loader
# 导入必要的模块，包括操作系统接口、日志记录和自定义的数据加载模块 Data_Loader。

# 这段代码提供了两个用于加载和处理数据集的函数，load_data 和 load_multiple_centralized_dataset。下面是对代码的逐行解释：
# 整体而言，这段代码提供了一种灵活的方式来加载和处理单个或多个数据集，适用于不同的训练模式，包括联邦学习和集中式训练。
# 通过 Data_Loader 类，代码封装了数据加载的详细逻辑，使得用户可以通过简单的函数调用来获取定制化的数据加载器。

# got it
def load_data(load_as, args=None, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.1, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default"):
    '''
    return:
    @train_date_num: means the number of training data from dataset totally, and usually the server training data number
    @test_data_num: means the number of testing data totally from dataset totally, and usually the server test data number
    @train_data_global: the DataLoader for Server and in this case is the whole dataset's training set
    @test_data_global: the DataLoader for Server and in this case is the whole dataset's testing set
    @data_local_num_dict: training set number of each clinet: {clinet_index: num_training_data}
    @train_data_dataloader_local_dict: Dict of DataLoader for each client: {clinet_index: Train_DataLoader_cliet_id}
    @test_data_dataloader_local_dict: Dict of DataLoader for each client: {clinet_index: Test_DataLoader_cliet_id}
    @class_num: total class number of the dataset
    @other_params: Dict for other parameters
    函数的多行字符串提供了返回值的详细说明。
    '''
    # 定义 load_data 函数，它接受多个参数以定制数据加载和处理过程。
    # datadir = get_new_datadir(args, datadir, dataset)
    other_params = {}

    data_loader = Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
        dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
        data_sampler,
        resize=resize, augmentation=augmentation, other_params=other_params)
    # 创建 Data_Loader 类的实例，传入所有必要的参数。
    train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl, train_data_local_num_dict, \
    test_data_local_num_dict, test_data_local_dl_dict, train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
    other_params = data_loader.load_data()  # FL mode load data
    # 调用 data_loader 的 load_data 方法执行数据加载，返回多个与数据集相关的值。
    return train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl,train_data_local_num_dict, \
        test_data_local_num_dict, test_data_local_dl_dict, train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
        other_params
    # 返回函数计算得到的数据集信息。


def load_multiple_centralized_dataset(load_as, args, process_id, mode, task,
                        dataset_list, datadir_list, batch_size, num_workers,
                        data_sampler=None,
                        resize=32, augmentation="default"):
    # 定义 load_multiple_centralized_dataset 函数，用于加载多个集中式数据集。
    train_dl_dict = {}
    test_dl_dict = {}
    train_ds_dict = {}
    test_ds_dict = {}
    class_num_dict = {}
    train_data_num_dict = {}
    test_data_num_dict = {}
    # 初始化多个字典来存储不同数据集的加载器、数据集对象、类别数和数据数量。
    for i, dataset in enumerate(dataset_list):
        # 遍历 dataset_list 中的每个数据集。
        # kwargs["data_dir"] = datadir_list[i]
        datadir = datadir_list[i]
        # 从 datadir_list 中获取当前数据集对应的数据目录。
        # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
        #     = load_centralized_data(load_as, args, process_id, mode, task,
        #                 dataset, datadir, batch_size, num_workers,
        #                 data_sampler=None,
        #                 resize=resize, augmentation=augmentation)
        train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
            = load_data(load_as=load_as, args=args, process_id=process_id,
                        mode="centralized", task="centralized",
                        dataset=dataset, datadir=datadir, batch_size=args.batch_size, num_workers=args.data_load_num_workers,
                        data_sampler=None,
                        resize=resize, augmentation=augmentation)
        # 对每个数据集调用 load_data 函数来加载数据。
        train_dl_dict[dataset] = train_dl
        test_dl_dict[dataset] = test_dl
        train_ds_dict[dataset] = other_params["train_ds"]
        test_ds_dict[dataset] = other_params["test_ds"]
        class_num_dict[dataset] = class_num
        train_data_num_dict[dataset] = train_data_num
        test_data_num_dict[dataset] = test_data_num
        # 将加载的数据加载器和其他信息存储到相应的字典中。
    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
        class_num_dict, train_data_num_dict, test_data_num_dict
    # 返回包含多个数据集信息的字典。












