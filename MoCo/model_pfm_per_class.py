from operator import imod
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from main_lincls import validate
# import utils
# import models
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import torch.nn.functional as F
import time
from main_lincls import AverageMeter, ProgressMeter, accuracy
from torchvision.models import resnet50
import torch
def evaluate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        probs = []
        targets = []
        for i, (images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            output = output[:,:10]
            # print(output)
            loss = criterion(output, target)
            prob = F.softmax(output, dim=1)

            probs.append(prob.detach())
            targets.append(target.detach())
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print(probs[:10], target[:10])
    return probs, targets

def model_pfm_per_class(model, num_classes, device, data_loader):
    # checkpoint = torch.load(checkpoint)
    # state_dict = checkpoint['model']
    # train_epoch = checkpoint['epoch']

    # model.load_state_dict(state_dict)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    probs, targets = evaluate(data_loader, model, criterion, device)
    probs_list = []
    for prob in probs:
        prob = prob.cpu().numpy().tolist()
        probs_list = probs_list + prob
    pred_list = []
    for prob in probs_list:
        prob = prob.index(max(prob))
        pred_list.append(prob)
    labels_list = []
    for target in targets:
        target = target.cpu().numpy().tolist()
        labels_list = labels_list + target
    print(classification_report(labels_list, pred_list, labels=list(range(num_classes))))
    
    return

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_size = 64
    n_workers = 4
    pin_mem = True

    DROP = 0.0
    DROP_PATH = 0.1

    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),    
        ]
    )
    # dataset_val, nb_classes = build_dataset(is_train=False, is_transform=False, data_set='CIFAR10')
    dataset_val = datasets.CIFAR10(root='/import/home/xliude/vs_projects/data', train=False, transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=b_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=False
    )

    ckp = torch.load('./CIFAR10_step001_linear/checkpoint_best.pth', map_location="cpu")['state_dict']
    # model = models.__dict__['resnet50']()
    # print(ckp.keys())
    model = resnet50()
    for k in list(ckp.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module') and not k.startswith('module.fc'):
                    # remove prefix
                    ckp[k[len("module."):]] = ckp[k]
                # delete renamed or unused k
                del ckp[k]
    model.load_state_dict(ckp, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    
    print("number of testing images:", len(dataset_val))
    model_pfm_per_class(model, 10, device, data_loader_val)