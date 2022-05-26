# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : 通用函数
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
from sklearn import metrics
import cv2


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


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, combiner, loss_f, optimizer, epoch_id, gpu, max_epoch):
        # print("data_loader length is {}".format(len(data_loader)))
        model.train()
        if combiner:
            combiner.update(epoch_id)

        conf_mat = np.zeros((9, 9))
        loss_sigma = []
        label_append = []
        outputs_append = []
        logits_append = []

        for i, (inputs, labels) in enumerate(data_loader):

            # inputs, labels = data
            # ones = torch.sparse.torch.eye(9) #用于把label转成one-hot编码
            # targets = ones.index_select(0, labels) #用于把label转成one-hot编码
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            # if mixup_fn is not None:
            #     inputs, labels = mixup_fn(inputs, labels)

            # labels = labels.long()
            outputs = model(inputs)

            if combiner:
                loss, now_acc = combiner.forward(model, loss_f, inputs, labels)

            else:
                loss = loss_f(outputs, labels)

            optimizer.zero_grad()
            # optimizer.module.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                # print(cate_i, pre_i)
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())
            logits_append.append(outputs.detach())

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 300 == 300 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))
            # print(labels)
            # break

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append, logits_append

    @staticmethod
    def mani_train(data_loader, model, optimizer, epoch_id, gpu, max_epoch):
        # print("data_loader length is {}".format(len(data_loader)))
        model.train()

        conf_mat = np.zeros((9, 9))
        loss_sigma = []
        label_append = []
        outputs_append = []
        logits_append = []

        for i, (inputs, labels) in enumerate(data_loader):

            # inputs, labels = data
            # ones = torch.sparse.torch.eye(9) #用于把label转成one-hot编码
            # targets = ones.index_select(0, labels) #用于把label转成one-hot编码
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            # if mixup_fn is not None:
            #     inputs, labels = mixup_fn(inputs, labels)

            labels = labels.long()

            outputs, loss = model(inputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())
            logits_append.append(outputs.detach())

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 300 == 300 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))
            # print(labels)
            # break

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append, logits_append

    @staticmethod
    def valid(data_loader, model, loss_f, gpu):
        model.eval()

        conf_mat = np.zeros((9, 9))
        loss_sigma = []
        label_append = []
        outputs_append = []
        logits_append = []
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):

                inputs, labels = data
                # ones = torch.sparse.torch.eye(9)
                # targets = ones.index_select(0, labels)
                # inputs, labels = inputs.to(device), labels.to(device)
                inputs, targets = inputs.cuda(gpu), labels.cuda(gpu)

                # targets = targets.cuda(gpu)

                outputs = model(inputs)
                loss = loss_f(outputs, targets)

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)

                # 统计混淆矩阵
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.

                # 统计loss
                loss_sigma.append(loss.item())

                # save labels and outputs to calculate ROC_auc and PR_auc
                probs = F.softmax(outputs, dim=1)
                label_append.append(labels.detach())
                outputs_append.append(probs.detach())
                logits_append.append(outputs.detach())
                # break

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append, logits_append

class SharpenImage(object):
    """Sharpen the inputted images"""
    def __init__(self, p=0.9):
        assert (isinstance(p, float))
        self.p = p
        self.kernel_sharpen_1 = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            image = np.array(img).copy()
            output_1 = cv2.filter2D(image, -1, self.kernel_sharpen_1)
            return Image.fromarray(output_1.astype('uint8'))
        else:
            return img






class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8'))
        else:
            return img



def cal_focal_loss_alpha(dataset):
    """
    for each class 计算 alpha 的值
    alpha = class num / overall class
    :param dataset
    :return: alpha of each class
    """
    combine_flag = False
    alpha = []
    if(type(dataset) == torch.utils.data.dataset.ConcatDataset):
        allClass = dataset.datasets[0].targets + dataset.datasets[1].targets
    else:
        allClass = dataset.targets

    labelList = list(set(allClass))
    for i in range(len(labelList)):
        ratio = round(allClass.count(labelList[i]) / len(allClass), 5)
        alpha.append(ratio)
    return alpha



def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(classes)):
            confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))



def cal_loss_eachClass(logtits, target, classNum):
    output_epoch_cat = torch.cat(logtits).cpu()
    target_epoch_cat = torch.cat(target).cpu()

    lossFun = nn.CrossEntropyLoss()
    loss_class = torch.zeros(classNum)
    for i in range(classNum):
        target_class = output_epoch_cat[target_epoch_cat == i]
        label_class = torch.mul(torch.ones(len(target_class)), i)
        label_class = label_class.type(torch.long)
        loss_class[i] = lossFun(target_class, label_class)
    return loss_class.numpy()



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
            label = int(row[-1].replace("\n", ""))
            # if label == 5:
            #     label = 2
            # else:
            #     label = 1
            labels.append(label)
        except ValueError as err:
            # print(row[0],row[1])
            print(' '.join(row[:-1]), row[-1])
    return file_names, labels

def dataset_info_no_others(txt_labels):
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
        if not int(row[-1].replace("\n", "")) == 1:
            file_names.append(' '.join(row[:-1]))
            try:
                # labels.append(int(row[1].replace("\n", "")))
                labels.append(int(row[-1].replace("\n", "")) - 1)
            except ValueError as err:
                # print(row[0],row[1])
                print(' '.join(row[:-1]), row[-1])
    return file_names, labels

#测试时读入others类，当作其他defect：marking defect
def dataset_info_test_others(txt_labels):
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
        if int(row[-1].replace("\n", "")) == 1:
            file_names.append(' '.join(row[:-1]))
            try:
                # labels.append(int(row[1].replace("\n", "")))
                labels.append(int(row[-1].replace("\n", "")))
            except ValueError as err:
                # print(row[0],row[1])
                print(' '.join(row[:-1]), row[-1])
        else:
            file_names.append(' '.join(row[:-1]))
            try:
                # labels.append(int(row[1].replace("\n", "")))
                labels.append(int(row[-1].replace("\n", "")) - 1)
            except ValueError as err:
                # print(row[0],row[1])
                print(' '.join(row[:-1]), row[-1])
    return file_names, labels

def dataset_info_only_others(txt_labels):
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
        if int(row[-1].replace("\n", "")) == 1:
            file_names.append(' '.join(row[:-1]))
            try:
                # labels.append(int(row[1].replace("\n", "")))
                labels.append(int(row[-1].replace("\n", "")))
            except ValueError as err:
                # print(row[0],row[1])
                print(' '.join(row[:-1]), row[-1])
        else:
            pass
    return file_names, labels


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' or mode == 'roc_auc' or mode == 'pr_auc' or mode == 'fpr' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()



def cal_auc(y_true_train, y_outputs_train, out_dir):
    """
    计算PR_AUC 和 ROC_AUC
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    y_true = torch.cat(y_true_train).cpu()
    y_score = torch.cat(y_outputs_train).cpu()
    roc_auc = metrics.roc_auc_score(y_true != 4, 1. - y_score[:, 4])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 4, 1. - y_score[:, 4])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 4, 1. - y_score[:, 4])
    fpr_980 = fpr[np.where(tpr >= 0.980)[0][0]]
    fpr_991 = fpr[np.where(tpr >= 0.991)[0][0]]
    fpr_993 = fpr[np.where(tpr >= 0.993)[0][0]]
    fpr_995 = fpr[np.where(tpr >= 0.995)[0][0]]
    fpr_997 = fpr[np.where(tpr >= 0.997)[0][0]]
    fpr_999 = fpr[np.where(tpr >= 0.999)[0][0]]
    fpr_1 = fpr[np.where(tpr == 1.)[0][0]]
    plt.plot(fpr, tpr, label='Valid')
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.title(out_dir)
    plt.show()
    # plt.savefig(os.path.join(out_dir, 'ROC.png'))
    plt.close()

    threshhold980 = thresholds[[np.where(tpr >= 0.980)[0][0]]]
    badScore_class_acc = evaluate_badScore(threshhold980, y_true, y_score)
    fpr_dict = {'fpr_980': fpr_980, 'fpr_991': fpr_991, 'fpr_993': fpr_993, 'fpr_995': fpr_995,
                'fpr_997': fpr_997, 'fpr_999': fpr_999, 'fpr_1': fpr_1, 'fpr_all': fpr, 'tpr_all': tpr, 'thresholds': thresholds, 'true_label': y_true, 'score': y_score}

    return roc_auc, pr_auc, fpr_dict, threshhold980, badScore_class_acc, 1. - y_score[:, 4]

def evaluate_badScore(threshhold980, y_label, y_score):
    # badScore_class = []
    badScore_class_acc = []
    badScore = 1. - y_score[:, 4]
    y_label_np = y_label.numpy()
    badScore_np = badScore.numpy()
    for i in range(len(set(list(y_label_np)))):
        badScore_class = badScore_np[y_label_np==i]
        # badScore_class_acc[i] = np.sum(badScore_class>=threshhold980) / len(badScore_class)
        badScore_class_acc.append(np.around(np.sum(badScore_class>=threshhold980) / len(badScore_class),2))
    return badScore_class_acc




def cal_auc_train(y_true_train, y_outputs_train):
    """
    计算PR_AUC 和 ROC_AUC
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    y_true = torch.cat(y_true_train).cpu()
    y_score = torch.cat(y_outputs_train).cpu()
    roc_auc = metrics.roc_auc_score(y_true != 4, 1. - y_score[:, 4])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 4, 1. - y_score[:, 4])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 4, 1. - y_score[:, 4])
    fpr_980 = fpr[np.where(tpr >= 0.98)[0][0]]
    fpr_991 = fpr[np.where(tpr >= 0.991)[0][0]]
    fpr_993 = fpr[np.where(tpr >= 0.993)[0][0]]
    fpr_995 = fpr[np.where(tpr >= 0.995)[0][0]]
    fpr_997 = fpr[np.where(tpr >= 0.997)[0][0]]
    fpr_999 = fpr[np.where(tpr >= 0.999)[0][0]]
    fpr_1 = fpr[np.where(tpr == 1.)[0][0]]

    fpr_dict = {'fpr_980': fpr_980, 'fpr_991': fpr_991, 'fpr_993': fpr_993, 'fpr_995': fpr_995,
                'fpr_997': fpr_997, 'fpr_999': fpr_999, 'fpr_1': fpr_1, 'thresholds': thresholds}

    return roc_auc, pr_auc, fpr_dict



def correct_label(orginal_name, original_label, name_correct):
    count = 0
    for i in range(len(name_correct)):
        for j in range(len(orginal_name)):
            if name_correct[i] in orginal_name[j]:
                count+=1
                # print("orginal_name is {} --- label is {}".format(orginal_name[j], original_label[j]))
                original_label[j] = 5
    # print("total count is {}".format(count))
    return original_label


def cal_fpr_auc_mean_std(var_list):
    '''
    :param var_list:
    :return: mean&std (best3, latest3, and overall)
    '''
    mean_list = []
    std_list = []
    mean_std_name = []

    var_np = np.array(var_list)
    mean_list.append(round(np.mean(var_np[:3]),2))
    mean_list.append(round(np.mean(var_np[3:]),2))
    mean_list.append(round(np.mean(var_np),2))

    std_list.append(round(np.std(var_np[:3]),3))
    std_list.append(round(np.std(var_np[3:]),3))
    std_list.append(round(np.std(var_np),3))

    mean_std_name = ['best3_mean_std', 'latest3_mean_std','overall_mean_std']

    return mean_list, std_list, mean_std_name


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
