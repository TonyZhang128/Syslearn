import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from zyn import AverageMeter
from data.data_loader import MNISTDataLoader

from options.train_options import TrainOptions
def train():
    loss = AverageMeter('loss')
def val():
    loss = AverageMeter('loss')
def test():
    ...

def main():
    #parse arguments
    opt = TrainOptions().parse()
    opt.device = torch.device("cuda")

    #construct data loader
    data_loader = MNISTDataLoader(opt)
    dataset_size = len(data_loader)

    #create validation set data loader if validation_on option is set
    if opt.validation_on:
        opt.mode = 'val'
        val_ds = dataset_builder.build(split='val')

    # Tensorboard start
    summary_writer = SummaryWriter()

    # Network Builders
    model_builder = ModelFactory(opt)
    model = model_builder.build(device = opt.device)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    # Set up loss functions
    train_criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()

    num_epoch = opt.num_epoch
    best_loss = float('inf')
    for epoch in range(num_epoch):
        start_time = time.time()
        train_loss = train(opt, epoch, train_ds, ds, summary_writer, model, optimizer, train_criterion)
        val_loss = val(opt, epoch, val_ds, ds, summary_writer, model, optimizer, val_criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s') 

        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)
        save_checkpoint(
            state_dict = model.module.state_dict(),
            is_best=is_best,
            folder=opt.ckpt,
            best_loss = best_loss
        )

if __name__ == '__main__':
    main()