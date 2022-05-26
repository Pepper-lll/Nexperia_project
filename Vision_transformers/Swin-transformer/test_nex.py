import sys
import os

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from sklearn import metrics
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from common_tools import cal_loss_eachClass, dataset_info, evaluate_badScore, Logger
from os.path import join, dirname
import os
from data.Nexperia_txt_dataset import textReadDataset
import pickle
from glob import glob
from prettytable import PrettyTable
import pandas as pd
from config import get_config
from models import build_model
from models.swin_transformer import SwinTransformer

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
    # print('y_true:',y_true)
    y_score = torch.cat(y_outputs_train).cpu()
    # print('y_score:', y_score)
    roc_auc = metrics.roc_auc_score(y_true != 4, 1. - y_score[:, 4])
    # roc_auc = metrics.roc_auc_score(y_true, y_score[:, 1])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 4, 1. - y_score[:, 4])
    # precision, recall, _ = metrics.precision_recall_curve(y_true, y_score[:, 1])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 4, 1. - y_score[:, 4])
    # fpr, tpr, thresholds = metrics.roc_curve(y_true != 1, y_score[:, 0])
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
    # badScore_class_acc = evaluate_badScore(threshhold980, y_true, y_score)
    fpr_dict = {'fpr_980': fpr_980, 'fpr_991': fpr_991, 'fpr_993': fpr_993, 'fpr_995': fpr_995,
                'fpr_997': fpr_997, 'fpr_999': fpr_999, 'fpr_1': fpr_1, 'fpr_all': fpr, 'tpr_all': tpr, 'thresholds': thresholds, 'true_label': y_true, 'score': y_score}

    return roc_auc, pr_auc, fpr_dict, threshhold980, 1 - y_score[:, 4]

def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    :param batch:
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

@torch.no_grad()
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
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            # targets = targets.cuda(gpu)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

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

            if(i % 300) == 0:
                print('~~~~~~~~~~~~~~~')

    acc_avg = conf_mat.trace() / conf_mat.sum()

    return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append, logits_append

if __name__ == "__main__":

    #++++++++ hyper parameters ++++++++#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###++++++++ for CE
    model_dir = '/import/home/xliude/vs_projects/Swin-Transformer/output/swin_tiny_patch4_window7_224/Nex_trainingset/default/ckpt_epoch_299.pth'
    data_dir = '/import/home/share/from_Nexperia_April2021/Feb2021/'
    backbone = "_Swin"
    # method = "_CE_weighedSampler"
    test_data_name = "_Jan"
    #++++++++ hyper parameters end ++++++++#

    name_test1, labels_test1 = dataset_info(join(data_dir, 'Feb2021_train_down.txt'))
    name_test2, labels_test2 = dataset_info(join(data_dir, 'Feb2021_val_down.txt'))
    name_test3, labels_test3 = dataset_info(join(data_dir, 'Feb2021_test_down.txt'))


    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + "_test" + "_" + "_".join([backbone, test_data_name])
    log_dir = os.path.join("/import/home/xliude/vs_projects/Swin-Transformer", "Nex_results", time_str)


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    class_names = (
    'Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip',
    'Chipping')
    # class_names = ('Defect', 'Pass')

    table = PrettyTable(['Model', 'Epoch', 'ACC', 'LOSS', 'FPR_980', 'FPR_995', 'FPR_100','AUC'])
    # model = build_model(config)
    model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=9,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=True)
    model.cuda()
    model.load_state_dict(torch.load(model_dir)['model'])
    table_loss_name = list(class_names)
    table_loss_name.insert(0, 'Model')
    table_loss = PrettyTable(table_loss_name)

    table_badScore_name = list(class_names)
    table_badScore_name.insert(0, 'Model')
    table_badScore_name.insert(1, 'Thr998')
    table_badScore = PrettyTable(table_badScore_name)
    table_loss = PrettyTable(['Model', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    num_classes = len(class_names)

    # MAX_EPOCH = 100     # 182     # 64000 / (45000 / 128) = 182 epochs
    BATCH_SIZE = 32
    # LR = 0.01
    CE = nn.CrossEntropyLoss()
    norm_mean = [0.5]
    norm_std = [0.5]

    valid_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data1 = textReadDataset(data_dir, name_test1, labels_test1, valid_transform)
    test_data2 = textReadDataset(data_dir, name_test2, labels_test2, valid_transform)
    test_data3 = textReadDataset(data_dir, name_test3, labels_test3, valid_transform)
    combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
    # 构建DataLoder
    test_loader = DataLoader(dataset=combined_data, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    print("")
    print("__________________________start testing____________________________")
    print("")
    loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid, logits_val = valid(test_loader, model, CE, device)

    roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, bad_score = cal_auc(y_true_valid, y_outputs_valid, log_dir)

    fpr_980_val = fpr_dict_val['fpr_980']
    fpr_990_val = fpr_dict_val['fpr_995']
    fpr_100_val = fpr_dict_val['fpr_1']
    loss_class_val = cal_loss_eachClass(logits_val, y_true_valid, num_classes)
    loss_class_val = np.around(loss_class_val,3)


    print("test Acc:{:.2%} test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%}".format(acc_valid, loss_valid, fpr_980_val, roc_auc_valid))
    print("loss of each class is:")
    print(["{0:0.3f} ".format(k) for k in list(loss_class_val)])



    # ============================ Put data into table ===========================
    table.add_row(['swin', round(acc_valid, 2), round(loss_valid, 3),
                    round(fpr_980_val, 2), round(fpr_990_val, 2), round(fpr_100_val, 2), round(roc_auc_valid, 2)])

    # loss_class_list = list(loss_class_val)
    # loss_class_list.insert(0, model[:-4])
    # table_loss.add_row(loss_class_list)

    print(table)
    print(table_loss)
    print(table_badScore)