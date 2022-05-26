import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')

class textReadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rootdir, names, labels, img_transformer=None):
        # self.data_path = join(dirname(__file__),'kfold')
        self.rootdir = rootdir
        self.names = names
        self.labels = labels
        # self.N = len(self.names)
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()

        img_name = self.rootdir + '/' + self.names[index]

        try:
            image = Image.open(img_name).convert('RGB')
        except:
            print(img_name)
            return None
        return self._image_transformer(image), int(self.labels[index] - 1)

    def get_labels(self):
        return self.labels

