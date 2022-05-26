import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
# import torchvision.models as models
import torch
from moco.imbalance_cifar import IMBALANCECIFAR10
from analyze_collapse import neural_collapse_embedding
from torchvision import models

if __name__ == '__main__':
    ckp = torch.load('./CIFAR10_step001/checkpoint.pth', map_location="cpu")['state_dict']
    # model = models.__dict__['resnet50']()
    model = models.resnet50()
    for k in list(ckp.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        ckp[k[len("module.encoder_q."):]] = ckp[k]
                    # delete renamed or unused k
                    del ckp[k]
    model.load_state_dict(ckp)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    dataset = IMBALANCECIFAR10(root='/import/home/xliude/vs_projects/data', imb_type='step', imb_factor=0.01,
                                        rand_number=0, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("start calculating!")
    equal_norm_activation, equa_ang, equa_ang_2, intra_v, intra_v_large, intra_v_small = neural_collapse_embedding(10, 5, model, dataloader, device)
    print('intra V / inter v:', intra_v)
    print('large intra V / inter v:', intra_v_large)
    print('small intra V / inter v:', intra_v_small)