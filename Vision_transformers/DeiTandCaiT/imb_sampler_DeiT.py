import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from model_trainer import train_one_epoch, evaluate
from losses import DistillationLoss, FocalLoss
# from samplers import RASampler
from torchsampler import ImbalancedDatasetSampler
import Deit_models
import utils
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 3
    seed_everything(seed)

    DIATRIBUTED_MODE = 'none'

    # MODEL = 'deit_small_patch16_224'
    MODEL = 'cait_XXS24_224'
    TEACHER_MODEL = 'regnety_160'
    TEACHER_PATH = ''
    DIST_TYPE = ['none', 'soft', 'hard']
    DIST_ALPHA = 0.5
    DIST_TAU = 1.0

    CLIP_GRAD = None
    FINETUNE = ''

    b_size = 64
    n_workers = 4
    pin_mem = True
    epochs = 400


    DROP = 0.0
    DROP_PATH = 0.1
    MODEL_EMA = [True, False]
    MODEL_EMA_DECAY = 0.99996

    MIXUP = 0.8 #0.8

    MIXCUT = 1.0
    MIXCUT_MINMAX = None
    MIXUP_PROB = 1.0
    MIXUP_SWITCH_PROB = 0.5
    MIXUP_MODE = ['batch', 'pair', 'elem']

    LABEL_SMOOTHING = 0.0 #0.1



    #optimizer
    args_opt = SimpleNamespace()
    args_opt.weight_decay = 0.05
    args_opt.lr = 5e-4
    args_opt.opt = 'adamw'  # 'lookahead_adam' to use `lookahead`
    args_opt.momentum = 0.9

    #learning rate schedule
    args_lrs = SimpleNamespace()
    args_lrs.epochs = epochs
    args_lrs.sched = 'cosine'
    args_lrs.lr = 5e-4
    args_lrs.warmup_lr = 1e-6
    args_lrs.min_lr = 1e-5
    args_lrs.decay_epochs = 30
    args_lrs.warmup_epochs = 5
    args_lrs.cooldown_epochs = 10
    args_lrs.patience_epochs = 10
    args_lrs.decay_rate = 0.1


    # dataset_train, nb_classes = build_dataset(is_train=True, is_transform=True, data_set='CIFAR10')
    # # dataset_val, _ = build_dataset(is_train=False, is_transform=False, data_set='CIFAR10')
    dataset_train, nb_classes = build_dataset(is_train=True, is_transform=True, data_set='Nex_trainingset')
    dataset_val, _ = build_dataset(is_train=False, is_transform=False, data_set='Nex_trainingset')
    # dataset_train, nb_classes = build_dataset(is_train=True, is_transform=True, data_set='IMBALANCECIFAR10')
    # dataset_val, _ = build_dataset(is_train=False, is_transform=False, data_set='CIFAR10')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=ImbalancedDatasetSampler(dataset_train),
        batch_size=b_size,
        # shuffle = True,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=b_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=False
    )
    mixup_fn = Mixup(
        mixup_alpha=MIXUP, cutmix_alpha=MIXCUT,
        cutmix_minmax=MIXCUT_MINMAX if MIXCUT_MINMAX else None,
        prob=MIXUP_PROB, switch_prob=MIXUP_SWITCH_PROB,
        mode=MIXUP_MODE[0], label_smoothing=LABEL_SMOOTHING,
        num_classes=nb_classes
    ) if MIXUP_PROB > 0.0 else None

    print(f"Creating model: {MODEL}")
    model = create_model(
        MODEL,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=DROP,
        drop_path_rate=DROP_PATH,
        drop_block_rate=None,
    )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_ema = None
    if MODEL_EMA[0]:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=MODEL_EMA_DECAY,
            # device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    optimizer = create_optimizer(args_opt, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args_lrs, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    # criterion = FocalLoss(class_num=nb_classes)

    if MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = FocalLoss(class_num=nb_classes)
    teacher_model = create_model(
        TEACHER_MODEL,
        pretrained=False,
        num_classes=nb_classes,
        global_pool='avg',
    )
    criterion = DistillationLoss(criterion, teacher_model, DIST_TYPE[0], DIST_ALPHA, DIST_TAU)


    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    time_str = time_str + "_CeiT" + "_imbCIFAR01_400epochs"
    output_dir = os.path.join("/import/home/xliude/PycharmProjects/nex_project/multi_classes/DeiT", "results", time_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = Path(output_dir)
    # output_dir = None
    max_acc = 0.0
    max_auc = 0.0
    for epoch in range(epochs):

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            CLIP_GRAD, model_ema, mixup_fn,
            set_training_mode=FINETUNE == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        if output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    # 'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_acc = max(max_acc, test_stats["acc1"])
        print(f'Max accuracy: {max_acc:.2f}%')

        print(f"Auc of the network on the {len(dataset_val)} test images: {test_stats['auc']:.1f}%")
        max_auc = max(max_auc, test_stats["auc"])
        print(f'Max auc: {max_auc:.4f}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


