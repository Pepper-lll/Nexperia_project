import sys
import os

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

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
from LDMA_train import NormedLinear

from os.path import join, dirname
import os
from Nexperia_txt_dataset import textReadDataset
import pickle
from glob import glob
from prettytable import PrettyTable
import pandas as pd
# ============================ Jasper added import-END============================

BASE_DIR = '/import/home/xliude/PycharmProjects/nex_project/multi_classes/CNN_models/'
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
    roc_auc = metrics.roc_auc_score(y_true != 3, 1. - y_score[:, 3])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 3, 1. - y_score[:, 3])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 3, 1. - y_score[:, 3])
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

    loss_sigma = []
    label_append = []
    outputs_append = []
    logits_append = []
    predicted_append = []

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


        # 统计loss
        loss_sigma.append(loss.item())

        # save labels and outputs to calculate ROC_auc and PR_auc
        probs = F.softmax(outputs, dim=1)
        label_append.append(labels.detach())
        outputs_append.append(probs.detach())
        logits_append.append(outputs.detach())
        predicted_append.append(predicted.detach())


    return np.mean(loss_sigma), label_append, outputs_append, logits_append, predicted_append

if __name__ == "__main__":

    #++++++++ hyper parameters ++++++++#

    ###++++++++ for CE
    model_dir = '07-17_21-49_Res50_LDAMloss'
    data_dir = '/import/home/share/from_Nexperia_April2021/Feb2021/'
    backbone = "_Res50"
    method = "_LDAM"
    test_data_name = "_Feb"
    #++++++++ hyper parameters end ++++++++#

    name_test1, labels_test1 = dataset_info_only_others(join(data_dir, 'Feb2021_train_down.txt'))
    name_test2, labels_test2 = dataset_info_only_others(join(data_dir, 'Feb2021_val_down.txt'))
    name_test3, labels_test3 = dataset_info_only_others(join(data_dir, 'Feb2021_test_down.txt'))



    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + "_test" + "_" + "_".join([backbone, method, test_data_name])
    log_dir = os.path.join("/import/home/xliude/PycharmProjects/nex_project/multi_classes/CNN_models", "results", "test",
                           time_str)


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)


    table = PrettyTable(['Model', 'Epoch', 'AUC'])


    # MAX_EPOCH = 100     # 182     # 64000 / (45000 / 128) = 182 epochs
    BATCH_SIZE = 64
    num_classes = 9
    # LR = 0.01



    # ============================ step 1/5 数据 ============================

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
    test_loader = DataLoader(dataset=combined_data, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # ============================ step 2/5 模型 ============================
    ###### ResNet50 #######
    resnet_model = models.resnet50()
    resnet_model.fc = NormedLinear(2048, num_classes)

    ###### Load Model #######
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


        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device)

        # ============================ step 3/5 损失函数 ============================
        criterion = nn.CrossEntropyLoss()
        # ============================ step 4/5 优化器 ============================

        # ============================ step 5/5 test ============================
        loss_valid,  y_true_valid, y_outputs_valid, logits_val, predicted_val = valid(test_loader, resnet_model,
                                                                                                     criterion, device)

        # roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, badScore_class_acc, bad_score = cal_auc(y_true_valid, y_outputs_valid, log_dir)


        # print("test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%} Epoch:{:.0f}".format(loss_valid, fpr_980_val, roc_auc_valid, train_epoch))
        badScore = 1. - y_outputs_valid[:, 4]
        print("read label:")
        print(y_true_valid)
        print("predicted label is")
        print(predicted_val)
        print("bad score:")
        print(badScore)



        # ============================ Put data into table ===========================
    #     table.add_row([model_name[:-4], train_epoch, round(loss_valid, 3), round(roc_auc_valid, 2)])
    #
    #
    #     # ============================ 保存测试结果 ============================
    #     test_results_dict = { "loss": loss_valid, "roc_auc_rec": roc_auc_valid, "fpr":fpr_dict_val}
    #     fileName = model_name[:-4] + "_results.pkl"
    #     test_results_file = open(join(log_dir + "/" + fileName), "wb")
    #     # pickle.dump(test_results_dict, test_results_file)
    #     test_results_file.close()
    #
    #     # ============================ save bad score ============================
    #
    #
    # print(table)
    print(" done ~~~~ {} ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M')))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f
