# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def cal_auc(y_true_train, y_outputs_train):
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
    # plt.plot(fpr, tpr, label='Valid')
    # plt.ylabel('tpr')
    # plt.xlabel('fpr')
    # plt.title(out_dir)
    # plt.show()
    # # plt.savefig(os.path.join(out_dir, 'ROC.png'))
    # plt.close()

    threshhold980 = thresholds[[np.where(tpr >= 0.980)[0][0]]]
    # badScore_class_acc = evaluate_badScore(threshhold980, y_true, y_score)
    fpr_dict = {'fpr_980': fpr_980, 'fpr_991': fpr_991, 'fpr_993': fpr_993, 'fpr_995': fpr_995,
                'fpr_997': fpr_997, 'fpr_999': fpr_999, 'fpr_1': fpr_1, 'fpr_all': fpr, 'tpr_all': tpr, 'thresholds': thresholds, 'true_label': y_true, 'score': y_score}

    return roc_auc, pr_auc, fpr_dict, threshhold980, 1. - y_score[:, 4]


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        targets = targets.long()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    probs_all = []
    target_all = []

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # print(target)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        probs = F.softmax(output, dim=1)
        probs_all.append(probs.detach())
        target_all.append(target.detach())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    auc = cal_auc(target_all, probs_all)[0]
    fpr_dict = cal_auc(target_all, probs_all)[2]
    fpr_980 = fpr_dict['fpr_980']
    metric_logger.meters['auc'].update(auc.item())
    metric_logger.meters['fpr_980'].update(fpr_980.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    #
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       'Auc {Auc.global_avg:.3f}  Fpr_980 {fpr.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss,
    #               Auc=metric_logger.auc, fpr=metric_logger.fpr_980))

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}