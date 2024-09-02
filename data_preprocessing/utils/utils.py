import numpy as np
import torch

# 这段代码提供了图像处理功能，包括一个图像裁剪类 Cutout 和几个用于加载图像的函数。以下是对代码中每个部分的详细解释：

class Cutout(object):
    # Cutout 类接受一个参数 length 来指定裁剪区域的大小。
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        # __call__ 方法实现了裁剪逻辑，随机选择图像中的一个区域并将该区域的像素值置为0，这通常用于数据增强。
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def accimage_loader(path):
    # accimage_loader 函数使用 accimage 库加载图像，如果加载失败则回退到 pil_loader。
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # pil_loader 函数使用 PIL（Python Imaging Library）库加载图像，并且将图像转换为 RGB 模式。
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    # default_loader 函数根据配置的图像后端选择使用 accimage_loader 或 pil_loader 来加载图像。
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


















