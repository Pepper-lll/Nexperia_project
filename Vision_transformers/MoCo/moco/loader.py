# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from .imbalance_cifar import SMALLCIFAR10,IMBALANCECIFAR10
from .Nexperia_txt_dataset import textReadDataset

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ImagePair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class ImagePair_Small(SMALLCIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

class ImagePair_IMB(IMBALANCECIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

class NexPair(textReadDataset):
    """Nextraining Dataset.
    """
    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()
        img_name = self.rootdir + '/' + self.names[index]
        try:
            img = Image.open(img_name).convert('RGB')
        except:
            print(img_name)
            return None

        if self._image_transformer is not None:
            im_1 = self._image_transformer(img)
            im_2 = self._image_transformer(img)

        return im_1, im_2