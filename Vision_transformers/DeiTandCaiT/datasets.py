import os
import json
import random

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from os.path import join, dirname
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from imbalance_cifar import IMBALANCECIFAR10
from rand_augment import RandAugment

class textReadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rootdir, names, labels, img_transformer=None):
        # self.data_path = join(dirname(__file__),'kfold')
        self.rootdir = rootdir
        self.names = names
        self.labels = labels
        # self.N = len(self.names)
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()

        img_name = self.rootdir + '/' + self.names[index]

        try:
            image = Image.open(img_name).convert('RGB')
        except:
            print(img_name)
            return None
        return self._image_transformer(image), int(self.labels[index] - 1)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# class AddPepperNoise(object):
#     """增加椒盐噪声
#     Args:
#         snr （float）: Signal Noise Rate
#         p (float): 概率值，依概率执行该操作
#     """
#
#     def __init__(self, snr, p=0.9):
#         assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
#         self.snr = snr
#         self.p = p
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): PIL Image
#         Returns:
#             PIL Image: PIL image.
#         """
#         if random.uniform(0, 1) < self.p:
#             img_ = np.array(img).copy()
#             h, w = img_.shape
#             signal_pct = self.snr
#             noise_pct = (1 - self.snr)
#             mask = np.random.choice((0, 1, 2), size=(h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
#             img_[mask == 1] = 255   # 盐噪声
#             img_[mask == 2] = 0     # 椒噪声
#             return Image.fromarray(img_.astype('uint8'))
#         else:
#             return img

def dataset_info(txt_labels):
    '''
    file_names:List, labels:List
    '''
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        # file_names.append(row[0])
        file_names.append(' '.join(row[:-1]))
        try:
            # labels.append(int(row[1].replace("\n", "")))
            labels.append(int(row[-1].replace("\n", "")))
        except ValueError as err:
            # print(row[0],row[1])
            print(' '.join(row[:-1]), row[-1])
    return file_names, labels

def build_dataset(is_train, is_transform, data_set):
    transform = build_transform(is_transform, data_set)

    if data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
        nb_classes = 10
    # elif args.data_set == 'IMNET':
    #     root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #     dataset = datasets.ImageFolder(root, transform=transform)
    #     nb_classes = 1000
    # elif args.data_set == 'INAT':
    #     dataset = INatDataset(args.data_path, train=is_train, year=2018,
    #                           category=args.inat_category, transform=transform)
    #     nb_classes = dataset.nb_classes
    # elif args.data_set == 'INAT19':
    #     dataset = INatDataset(args.data_path, train=is_train, year=2019,
    #                           category=args.inat_category, transform=transform)
    #     nb_classes = dataset.nb_classes

    elif data_set == 'IMBALANCECIFAR10':
        dataset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=0.1,
                                   rand_number=0, train=is_train, download=True, transform=transform)
        nb_classes = 10

    elif data_set == 'Nex_trainingset':
        data_dir = '/import/home/share/from_Nexperia_April2021/Nex_trainingset/'

        name_train, labels_train = dataset_info(join(data_dir, 'Nex_trainingset_train.txt'))
        name_val, labels_val = dataset_info(join(data_dir, 'Nex_trainingset_val.txt'))

        if is_train and is_transform:
            dataset = textReadDataset(data_dir, name_train, labels_train, transform)
        else:
            dataset = textReadDataset(data_dir, name_val, labels_val, transform)
        nb_classes = 9

    return dataset, nb_classes


def build_transform(is_transform, data_set):
    resize_im = 224
    if is_transform:
        # this should always dispatch to transforms_imagenet_train
        if data_set == 'Nex_trainingset':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=4, padding_mode='edge'),
                # SharpenImage(p=0.5),
                # AddPepperNoise(0.9, p=0.3),
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=4, shear=4, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                    transforms.RandomAffine(degrees=0),
                ]),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.7),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 2), value=(0)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            # transform.transforms.insert(0, RandAugment(3, 0.5))
            return transform
        else:
            transform = create_transform(
                input_size=224,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25, #Random Erase prb
                re_mode='pixel', #Random Erase mode
                re_count=1, #random erase count
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    224, padding=4)
            return transform

    if not is_transform:
        if data_set == 'Nex_trainingset':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            return transform
        else:
            t = []
            if resize_im:
                size = int((256 / 224) * 224)
                t.append(
                    transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(224))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            return transforms.Compose(t)
