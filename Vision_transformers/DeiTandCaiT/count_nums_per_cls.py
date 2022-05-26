import torch

from datasets import build_dataset


dataset_train, nb_classes = build_dataset(is_train=True, is_transform=True, data_set='IMBALANCECIFAR10')

data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle = True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
cls_dict = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0}
for i, data in enumerate(data_loader_train):
    _, labels = data
    # print(labels)
    for idx in range(labels.size(-1)):
        # print(idx)
        # print(labels[idx].item())
        cls_dict[str(labels[idx].item())]+=1

print(cls_dict)
