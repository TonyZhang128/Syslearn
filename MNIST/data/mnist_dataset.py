import torch
import os
import torch.utils.data as data
from torchvision import transforms
class MNISTDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.name = opt.dataset
        self.root = opt.dataset_path
        self.mode = opt.mode
        if opt.transform:
            self.transform = transforms.ToTensor 
        if self.mode == 'train':
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.mode, )
            )
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError