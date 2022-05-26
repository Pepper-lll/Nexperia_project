import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from main_lincls import dataset_info
from moco.Nexperia_txt_dataset import textReadDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torchvision.models as models

@torch.no_grad()
def valid(data_loader, model, loss_f, gpu):
    model.eval()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    loss_sigma = []
    label_append = []
    outputs_append = []
    logits_append = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().float()
            total += len(labels)
            # 统计loss
            loss_sigma.append(loss.item())
            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())
            logits_append.append(outputs.detach())
            if(i % 300) == 0:
                print('~~~~~~~~~~~~~~~')
    acc_avg = (correct/total).cpu().detach().data.numpy()
    return np.mean(loss_sigma), acc_avg, label_append, outputs_append, logits_append

def cal_auc(y_true_train, y_outputs_train):
    """
    计算PR_AUC 和 ROC_AUC
    """
    y_true = torch.cat(y_true_train).cpu()
    y_score = torch.cat(y_outputs_train).cpu()
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

    threshhold980 = thresholds[[np.where(tpr >= 0.980)[0][0]]]
    # badScore_class_acc = evaluate_badScore(threshhold980, y_true, y_score)
    fpr_dict = {'fpr_980': fpr_980, 'fpr_991': fpr_991, 'fpr_993': fpr_993, 'fpr_995': fpr_995,
                'fpr_997': fpr_997, 'fpr_999': fpr_999, 'fpr_1': fpr_1, 'fpr_all': fpr, 'tpr_all': tpr, 'thresholds': thresholds, 'true_label': y_true, 'score': y_score}

    return roc_auc, pr_auc, fpr_dict, threshhold980, 1 - y_score[:, 4]

def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    model = models.__dict__['resnet50']()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 9)
    ckp = torch.load('./finetune_NEX_subset05/checkpoint_0049.pth', map_location="cpu")
    print('epoch of the model:', ckp['epoch'])
    state_dict = ckp['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    valid_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    months = ['Feb', 'Mar']
    # months = ['Mar']
    new_m = ['Apr', 'May', 'Jun']
    for month in months:
        print('Now testing', month, 'data:')
        data_dir = '/import/home/share/from_Nexperia_April2021/' + month + '2021/'
        name_test1, labels_test1 = dataset_info(join(data_dir, month + '2021_train_down.txt'))
        name_test2, labels_test2 = dataset_info(join(data_dir, month + '2021_val_down.txt'))
        name_test3, labels_test3 = dataset_info(join(data_dir, month + '2021_test_down.txt'))
        test_data1 = textReadDataset(data_dir, name_test1, labels_test1, valid_transform)
        test_data2 = textReadDataset(data_dir, name_test2, labels_test2, valid_transform)
        test_data3 = textReadDataset(data_dir, name_test3, labels_test3, valid_transform)
        combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
        # 构建DataLoder
        test_loader = DataLoader(dataset=combined_data, collate_fn= collate_fn, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        loss_valid, acc_valid, y_true_valid, y_outputs_valid, logits_val = \
            valid(test_loader, model, nn.CrossEntropyLoss(), device)
        roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, bad_score = \
            cal_auc(y_true_valid, y_outputs_valid)
        fpr_980_val = fpr_dict_val['fpr_980']
        print('Result of', month, 'data:')
        print("test Acc:{:.2%} test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%}".format(acc_valid, loss_valid, fpr_980_val, roc_auc_valid))
    for month in new_m:
        print('Now testing', month, 'data:')
        data_dir = '/import/home/share/SourceData/' + month + '2021/'
        name_test1, labels_test1 = dataset_info(join(data_dir, month + '2021_train_down.txt'))
        name_test2, labels_test2 = dataset_info(join(data_dir, month + '2021_val_down.txt'))
        name_test3, labels_test3 = dataset_info(join(data_dir, month + '2021_test_down.txt'))
        test_data1 = textReadDataset(data_dir, name_test1, labels_test1, valid_transform)
        test_data2 = textReadDataset(data_dir, name_test2, labels_test2, valid_transform)
        test_data3 = textReadDataset(data_dir, name_test3, labels_test3, valid_transform)
        combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
        # 构建DataLoder
        test_loader = DataLoader(dataset=combined_data, collate_fn= collate_fn, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        loss_valid, acc_valid, y_true_valid, y_outputs_valid, logits_val = \
            valid(test_loader, model, nn.CrossEntropyLoss(), device)
        roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, bad_score = \
            cal_auc(y_true_valid, y_outputs_valid)
        fpr_980_val = fpr_dict_val['fpr_980']
        print('Result of', month, 'data:')
        print("test Acc:{:.2%} test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%}".format(acc_valid, loss_valid, fpr_980_val, roc_auc_valid))
        