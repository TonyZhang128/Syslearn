import torch.utils.data as data


class MNISTDataLoader():
    def __init__(self):
        self.name = 'MNIST'
    
    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        if opt.mode == 'train':
            self.dataloader = data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.nThreads,
                collate_fn=collate
            )
        
        elif opt.mode == 'val':
            self.dataloader = data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.nThreads,
                collate_fn=collate
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