import torch

model = torch.load('/import/home/xliude/PycharmProjects/nex_project/multi_classes/ResT-main/results/08-14_21-22_ResTnex_trainset/checkpoint.pth',map_location='cpu')
print(model.keys())
print(model['state_dict'])