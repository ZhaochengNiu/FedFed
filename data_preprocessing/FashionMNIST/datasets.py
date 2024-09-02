import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST
# 导入了日志记录、NumPy、PyTorch数据加载工具、PIL图像处理库、FashionMNIST数据集类等。
import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout

# 这段代码定义了用于加载和处理Fashion-MNIST数据集的函数和类，包括数据变换、数据集的子集加载以及自定义的数据集类。
# 以下是对代码中每个部分的详细解释：
# 整体来看，这段代码提供了一套工具来加载、变换和处理Fashion-MNIST数据集，支持自定义的数据变换和数据集子集操作。
# 代码中使用了日志记录来跟踪数据处理的进度和状态，并考虑了不同的数据集使用场景。
# 此外，通过自定义的PyTorch数据集类，可以灵活地处理各种类型的图像数据。


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 配置日志记录系统，设置日志级别为INFO。
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
# 定义了支持的图像文件扩展名。


def data_transforms_fmnist(resize=28, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    # 根据提供的参数，定义Fashion-MNIST数据集的训练和测试变换操作。
    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        pass
    else:
        raise NotImplementedError

    if resize == 28:
        pass
    else:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        #train_transform.transforms.append(RandAugmentMC(n=2, m=10))
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.ToTensor())

    return None, None, train_transform, test_transform


class FashionMNIST_truncated(data.Dataset):
    # 继承自torch.utils.data.Dataset，用于加载Fashion-MNIST数据集的一个子集。
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        # 初始化自定义数据集类，接受数据根目录、数据索引、训练/测试标志、变换操作等参数。
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # 用于构建数据集，根据dataidxs参数选择子集。
        print("download = " + str(self.download))
        mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # print("train member of the class: {}".format(self.train))
        # data = cifar_dataobj.train_data
        data = mnist_dataobj.data
        targets = mnist_dataobj.targets
        # targets = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        # 根据索引获取数据集中的一项（图像和标签）。
        img, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        # 返回数据集的长度。
        return len(self.data)


class FashionMNIST_truncated_WO_reload(data.Dataset):
    # 类似于FashionMNIST_truncated，但使用已加载的完整数据集对象来避免重复加载。
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # 使用完整数据集对象构建子集。
        # print("download = " + str(self.download))
        # mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
            data = self.full_dataset.data[self.dataidxs]
            targets = np.array(self.full_dataset.targets)[self.dataidxs]
        else:
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)


        return data, targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)



