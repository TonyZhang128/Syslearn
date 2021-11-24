import torch
from torch.utils.data import DataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset == 'MNIST':
        from data.mnist_dataset import MNISTDataset
        dataset = MNISTDataset(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset)

    print("dataset [%s] was created" % (dataset.name))
    return dataset

def collate(bacth):
    return [collate(samples) for samples in bacth]

# class MNISTDataLoader(data.dataloader):
class MNISTDataLoader():
    def __init__(self, opt):
        self.name = 'MNIST'
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.collate = collate
        if opt.mode == 'train':
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.nThreads,
                # collate_fn=self.collate
            )
        
        elif opt.mode == 'test':
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.nThreads,
                # collate_fn=self.collate
            )

    def load_data():
        return None
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def CreateDataLoader(opt):
    from data.mnist_dataset import mnist_dataset