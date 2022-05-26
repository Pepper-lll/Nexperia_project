import sys
import os

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

from sklearn import metrics
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from common_tools import ModelTrainer, show_confMat, plot_line, \
    dataset_info_no_others, dataset_info_test_others, dataset_info_only_others, \
    cal_loss_eachClass, dataset_info, evaluate_badScore, Logger

# ============================ Jasper added import============================
from torchvision import models
from timm.models import create_model
from timm.models.registry import register_model

from os.path import join, dirname
import os
from Nexperia_txt_dataset import textReadDataset
import pickle
from glob import glob
from prettytable import PrettyTable
import pandas as pd
# ============================ Jasper added import-END============================

BASE_DIR = '/import/home/xliude/PycharmProjects/nex_project/multi_classes/DeiT/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return roc_auc, pr_auc, fpr_dict, threshhold980, badScore_class_acc, 1. - y_score[:, 3]

############ Jasper added to process the empty images ###################
def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    :param batch:
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def valid(data_loader, model, loss_f, gpu):
    model.eval()

    conf_mat = np.zeros((9, 9))
    loss_sigma = []
    label_append = []
    outputs_append = []
    logits_append = []

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

    acc_avg = conf_mat.trace() / conf_mat.sum()

    return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append, logits_append

if __name__ == "__main__":

    #++++++++ hyper parameters ++++++++#

    ###++++++++ for CE
    model_dir = '08-15_21-33_CeiT_nextrainingset'
    data_dir = '/import/home/share/from_Nexperia_April2021/Feb2021/'
    test_data = 'Feb2021'
    model = "_CaiT"
    method = "_focal"
    test_data_name = "_Feb"
    #++++++++ hyper parameters end ++++++++#

    name_test1, labels_test1 = dataset_info(join(data_dir, 'Feb2021_train_down.txt'))
    name_test2, labels_test2 = dataset_info(join(data_dir, 'Feb2021_val_down.txt'))
    name_test3, labels_test3 = dataset_info(join(data_dir, 'Feb2021_test_down.txt'))



    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + "_test" + "_" + "_".join([model, method, test_data])
    log_dir = os.path.join("/import/home/xliude/PycharmProjects/nex_project/multi_classes/DeiT", "results", "test",
                           time_str)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    class_names = (
    'Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip',
    'Chipping')

    table = PrettyTable(['Model', 'Epoch', 'ACC', 'LOSS', 'FPR_980', 'FPR_995', 'FPR_100','AUC'])

    table_loss_name = list(class_names)
    table_loss_name.insert(0, 'Model')
    table_loss = PrettyTable(table_loss_name)

    table_badScore_name = list(class_names)
    table_badScore_name.insert(0, 'Model')
    table_badScore_name.insert(1, 'Thr998')
    table_badScore = PrettyTable(table_badScore_name)
    # table_loss = PrettyTable(['Model', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    num_classes = len(class_names)

    # MAX_EPOCH = 100     # 182     # 64000 / (45000 / 128) = 182 epochs
    BATCH_SIZE = 32
    # LR = 0.01
    ############ Nexperial_compare_traingSet Train set ############
    norm_mean = [0.5]
    norm_std = [0.5]

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data1 = textReadDataset(data_dir, name_test1, labels_test1, valid_transform)
    test_data2 = textReadDataset(data_dir, name_test2, labels_test2, valid_transform)
    test_data3 = textReadDataset(data_dir, name_test3, labels_test3, valid_transform)
    combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
    # 构建DataLoder
    test_loader = DataLoader(dataset=combined_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    MODEL = 'cait_xxs24_224'
    model = create_model(
        MODEL,
        pretrained=False,
        num_classes=9,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    model.to(device)
    saved_model_dir = os.path.join(BASE_DIR, "results", model_dir)
    os.chdir(saved_model_dir) #os.chdir 改变当前路径为()
    print("Current working directory: {0}".format(os.getcwd()))

    for model_name in sorted(glob("*.pth")):
        checkpoint_file_loc = os.path.join(saved_model_dir, model_name)
        print("")
        print("________________________________________________"+model_name)
        print("")

        checkpoint = torch.load(checkpoint_file_loc)
        state_dict = checkpoint['model']
        train_epoch = checkpoint['epoch']


        model.load_state_dict(state_dict)
        model.to(device)

        # ============================ step 3/5 损失函数 ============================
        criterion = nn.CrossEntropyLoss()
        # ============================ step 4/5 优化器 ============================

        # ============================ step 5/5 test ============================
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid, logits_val = valid(test_loader, model, criterion, device)

        roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, badScore_class_acc, bad_score = cal_auc(y_true_valid, y_outputs_valid, log_dir)

        fpr_980_val = fpr_dict_val['fpr_980']
        fpr_990_val = fpr_dict_val['fpr_995']
        fpr_100_val = fpr_dict_val['fpr_1']
        loss_class_val = cal_loss_eachClass(logits_val, y_true_valid, num_classes)
        loss_class_val = np.around(loss_class_val,3)


        print("test Acc:{:.2%} test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%} Epoch:{:.0f}".format(acc_valid, loss_valid, fpr_980_val, roc_auc_valid, train_epoch))
        print("loss of each class is:")
        print(["{0:0.3f} ".format(k) for k in list(loss_class_val)])



        # ============================ Put data into table ===========================
        table.add_row([model_name[:-4], train_epoch, round(acc_valid, 2), round(loss_valid, 3),
                       round(fpr_980_val, 2), round(fpr_990_val, 2), round(fpr_100_val, 2), round(roc_auc_valid, 2)])

        loss_class_list = list(loss_class_val)
        loss_class_list.insert(0, model_name[:-4])
        table_loss.add_row(loss_class_list)

        table_badScore_list = badScore_class_acc
        table_badScore_list.insert(0, model_name[:-4])
        table_badScore_list.insert(1, threshhold980)
        table_badScore.add_row(table_badScore_list)

        # ============================ 保存测试结果 ============================
        test_results_dict = {"acc": acc_valid, "loss": loss_valid, "roc_auc_rec": roc_auc_valid, "fpr":fpr_dict_val, "confMat":mat_valid, "loss_class":loss_class_val}
        fileName = model_name[:-4] + "_results.pkl"
        test_results_file = open(join(log_dir + "/" + fileName), "wb")
        pickle.dump(test_results_dict, test_results_file)
        test_results_file.close()

        show_confMat(mat_valid, class_names, model_name, log_dir, verbose=True)

    print(table)
    print(table_loss)
    print(table_badScore)
    print(" done ~~~~ {} ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M')))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    # f = open(os.path.join(log_dir, 'log.txt'), 'a')
    # sys.stdout = f
    # sys.stderr = f