# -*- coding: utf-8 -*-
"""
# @file name  : train_resnet_focal.py
# @author     : Jasper
# @date       : 2021-03-01
# @brief      : resnet training on Nexperia dataset
"""
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from common_tools import ModelTrainer, show_confMat, \
    plot_line, cal_auc_train, cal_focal_loss_alpha, cal_loss_eachClass,\
    dataset_info,  dataset_info_no_others, set_seed, correct_label

# ============================ Jasper added import============================
from torchvision import models
from bisect import bisect_right
from Nexperia_txt_dataset import textReadDataset
from os.path import join, dirname
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from LDAM_loss import LDAMLoss
# from tools import my_resnet
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
# ============================ Jasper added import-END============================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", '3')
# device = torch.device(f'cuda:{3}')

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def save_model(epoch, auc, fpr_dict, _save_models_dir, model_dict):
    # tmp_auc, tmp_fpr_980 = auc_dict['auc'], auc_dict['fpr_980']
    tmp_auc = auc
    for i,rec in enumerate(topk):
        if tmp_auc > rec:
            for j in range(len(topk)-1,i,-1):
                topk[j] = topk[j-1]
                _j, _jm1 = join(_save_models_dir, f"_best{j+1}.pth"),\
                join(_save_models_dir, f"_best{j}.pth")
                if  os.path.exists(_jm1):
                    os.rename(_jm1,_j)
            topk[i] = tmp_auc
            model_saved_path = join(_save_models_dir, f"_best{i+1}.pth")
            state_to_save = {'model':model_dict, 'auc_dict':auc,'fpr_dict':fpr_dict, 'epoch':epoch}
            torch.save(state_to_save, model_saved_path)
            print(f'=>Best{i+1} model updated')
            break

    if epoch in range(MAX_EPOCH-3, MAX_EPOCH):
        model_saved_path = join(_save_models_dir, f"_epochs{epoch}.pth")
        state_to_save = {'model': model_dict, 'auc_dict': auc,'fpr_dict':fpr_dict, 'epoch': epoch}
        torch.save(state_to_save, model_saved_path)
        # print(f'=>Last{MAX_EPOCH - epoch} model updated')


class Logger(object):
    """
    save the output of the console into a log file
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    set_seed(1)  # 设置随机种子

    cls2int = {
        'Others': 1,
        'Marking_defect': 2,
        'Lead_glue': 3,
        'Lead_defect': 4,
        'Pass': 5,
        'Foreign_material': 6,
        'Empty_pocket': 7,
        'Device_flip': 8,
        'Chipping': 9,
    }
    class_l = list(cls2int.keys())

    #++++++++ hyper parameters ++++++++#
    data_dir = '/import/home/share/from_Nexperia_April2021/Nex_trainingset/'
    backbone = "_Res50"
    method = "_LDAMloss_DRW_06"
    data_name = "_NexTrainSet"
    # additional_info = '_grayInput_transformSet_SharpPepperRand'


    name_train, labels_train = dataset_info(join(data_dir, 'Nex_trainingset_train.txt'))
    name_val, labels_val = dataset_info(join(data_dir, 'Nex_trainingset_val.txt'))


    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + backbone + method
    log_dir = os.path.join("/import/home/xliude/PycharmProjects/nex_project/multi_classes/CNN_models", "results",
                           time_str)
    log_dir_train = os.path.join(log_dir, "train")
    log_dir_val = os.path.join(log_dir, "val")
    log_dir_tensorboard = os.path.join(log_dir, "tensorboard")

    writer = SummaryWriter(log_dir=log_dir_tensorboard, comment='_scalars', filename_suffix="12345678")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_val):
        os.makedirs(log_dir_val)
    if not os.path.exists(log_dir_tensorboard):
        os.makedirs(log_dir_tensorboard)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    # class_names = ('chipping', 'device_flip', 'empty_pocket', 'foreign_material', 'good', 'lead_defect', 'lead_glue', 'marking_defect')
    class_names = ('Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip', 'Chipping')

    num_classes = len(class_names)

    MAX_EPOCH = 200
    BATCH_SIZE = 32
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [100, 150]  # divide it by 10 at 32k and 48k iterations
    topk = [0 for _ in range(3)]
    # ============================ step 1/5 数据 ============================
    # norm_mean = [0.2391]
    # norm_std = [0.1365]

    norm_mean = [0.5]
    norm_std = [0.5]

    # train_transform = transforms.Compose([
    #     # transforms.Grayscale(1),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     # transforms.RandomResizedCrop(224),
    #     transforms.ColorJitter(0.2, 0.2, 0.2),
    #     transforms.RandomAffine(degrees=5, translate=(0.15, 0.1), scale=(0.75, 1.05)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(norm_mean, norm_std),
    # ])

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4, padding_mode = 'edge'),
        # SharpenImage(p=0.5),
        # AddPepperNoise(0.9, p=0.3),
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=4, shear=4, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.RandomAffine(degrees=0),
        ]),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.7),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 2), value=(0)),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # Construct Dataset
    train_data = textReadDataset(data_dir, name_train, labels_train, train_transform)
    valid_data = textReadDataset(data_dir, name_val, labels_val, valid_transform)
    # Construct Dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=False)


    # ============================ step 2/5 模型 ============================

    ######ResNet34#######
    resnet_model = models.resnet50()
    resnet_model.fc = NormedLinear(2048, num_classes)
    resnet_model.to(device)

    # ============================ step 3/5 loss function ============================
    # weight = None
    weight = 'DRW'
    cls_num_list = [2, 1830, 748, 532, 39446, 3399, 7527, 166, 10383]
    # criterion = LDAMLoss(cls_num_list=cls_num_list)
    # alphaTensor = torch.tensor(alpha)
    # criterion = FocalLoss(class_num=num_classes, alpha=None, gamma=2)
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # Optimizer

    # scheduler = WarmupMultiStepLR(optimizer=optimizer,
    #                               milestones=milestones,
    #                               gamma=0.1,
    #                               warmup_factor=0.1,
    #                               warmup_iters=5,
    #                               warmup_method="linear",
    #                               last_epoch=-1)


# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    loss_class_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    roc_auc_rec = {"train": [], "valid": []}
    # pr_auc_rec = {"train": [], "valid": []}
    fpr_980_rec = {"train": [], "valid": []}
    fpr_rec = {"train": [], "valid": []}
    confMat_rec = {"train": [], "valid": []}
    best_acc, best_epoch, best_auc = 0, 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):
        adjust_learning_rate(optimizer, epoch, LR)

        if weight == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        elif weight == None:
            train_sampler = None
            per_cls_weights = None

        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)


        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train, y_true_train, y_outputs_train, logits_train = ModelTrainer.train(train_loader, resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid, logits_val = ModelTrainer.valid(valid_loader, resnet_model, criterion, device)

        roc_auc_train, pr_auc_train, fpr_dict_train = cal_auc_train(y_true_train, y_outputs_train)
        roc_auc_valid, pr_auc_valid, fpr_dict_val = cal_auc_train(y_true_valid, y_outputs_valid)

        fpr_980_train = fpr_dict_train['fpr_980']
        fpr_980_val = fpr_dict_val['fpr_980']

        # print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr:{:.2%} Valid fpr:{:.2%} LR:{}".format(
        #     epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, fpr_train, fpr_valid, optimizer.param_groups[0]["lr"]))
        print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr:{:.2%} Valid fpr:{:.2%} Train AUC:{:.2%} Valid AUC:{:.2%} LR:{}".format(
                epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, fpr_980_train, fpr_980_val, roc_auc_train, roc_auc_valid, optimizer.param_groups[0]["lr"]))


        loss_class_train = cal_loss_eachClass(logits_train, y_true_train, num_classes)
        loss_class_val = cal_loss_eachClass(logits_val, y_true_valid, num_classes)
        loss_class_train_dict = {}
        loss_class_val_dict = {}
        for i in range(len(class_l)):
            loss_class_train_dict[class_l[i]] = loss_class_train[i]
            loss_class_val_dict[class_l[i]] = loss_class_val[i]

        ###############---write to tensorboard for visualization
        writer.add_scalars("Loss", {"Train": loss_train, "Valid": loss_valid}, epoch)
        writer.add_scalars("Accuracy", {"Train": acc_train, "Valid": acc_valid}, epoch)
        writer.add_scalars("FPR", {"Train": fpr_980_train, "Valid": fpr_980_val}, epoch)
        writer.add_scalars("AUC", {"Train": pr_auc_train, "Valid": pr_auc_valid}, epoch)
        writer.add_scalars("loss_each_class_train", loss_class_train_dict, epoch)
        writer.add_scalars("loss_each_class_val", loss_class_val_dict, epoch)


        # scheduler.step()  # Update learning rate




        # Save
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        roc_auc_rec["train"].append(roc_auc_train), roc_auc_rec["valid"].append(roc_auc_valid)
        # pr_auc_rec["train"].append(pr_auc_train), pr_auc_rec["valid"].append(pr_auc_valid)
        fpr_980_rec["train"].append(fpr_980_train), fpr_980_rec["valid"].append(fpr_980_val)
        # fpr_rec["train"].append(fpr_dict_train), fpr_rec["valid"].append(fpr_dict_val)
        confMat_rec["train"].append(mat_train), confMat_rec["valid"].append(mat_valid)
        loss_class_rec["train"].append(loss_class_train), loss_class_rec["valid"].append(loss_class_val)

        np.save(os.path.join(log_dir_train, 'loss_rec.npy'), loss_rec["train"])
        np.save(os.path.join(log_dir_train, 'acc_rec.npy'), acc_rec["train"])
        np.save(os.path.join(log_dir_train, 'roc_auc_rec.npy'), roc_auc_rec["train"])
        np.save(os.path.join(log_dir_train, 'fpr_980_rec.npy'), fpr_980_rec["train"])
        # np.save(os.path.join(log_dir_train, 'fpr_rec.npy'), fpr_rec["valid"])
        np.save(os.path.join(log_dir_train, 'confMat_rec.npy'), confMat_rec["train"])
        np.save(os.path.join(log_dir_train, 'loss_class_rec.npy'), loss_class_rec["train"])
        # np.save(os.path.join(log_dir_train, 'pr_auc_rec.npy'), pr_auc_rec["train"])

        np.save(os.path.join(log_dir_val, 'loss_rec.npy'), loss_rec["valid"])
        np.save(os.path.join(log_dir_val, 'acc_rec.npy'), acc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'roc_auc_rec.npy'), roc_auc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'fpr_980_rec.npy'), fpr_980_rec["valid"])
        # np.save(os.path.join(log_dir_val, 'fpr_rec.npy'), fpr_rec["valid"])
        np.save(os.path.join(log_dir_val, 'confMat_rec.npy'), confMat_rec["valid"])
        np.save(os.path.join(log_dir_val, 'loss_class_rec.npy'), loss_class_rec["valid"])
        # np.save(os.path.join(log_dir_val, 'pr_auc_rec.npy'), pr_auc_rec["valid"])

        if epoch > MAX_EPOCH/2-10:
            if best_auc < roc_auc_valid:
                best_auc = roc_auc_valid
                best_epoch = epoch


            save_model(epoch, roc_auc_valid, fpr_dict_val, log_dir, resnet_model.state_dict())



    print(" done ~~~~ {}, best auc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_auc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f
