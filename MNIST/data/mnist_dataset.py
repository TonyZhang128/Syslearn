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
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        else:
            self.transform = None
        if self.mode == 'train':
            self.data, self.labels = torch.load(
                os.path.join(self.root, 'training' + '.pt')
            )
        elif self.mode == 'test':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.mode+'.pt') 
            )
    def __getitem__(self, index):
        img, labels = self.data[index], int(self.labels[index])

        if self.transform:
            ...
            # img = self.transform(img)
        
        return img, labels  

    def __len__(self):
        return len(self.data)