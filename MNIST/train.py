import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from zyn import AverageMeter
from zyn.utils import epoch_time, save_checkpoint, accuracy
from data.data_loader import MNISTDataLoader
from models.mnist_model import MNISTModel, Net
from options.train_options import TrainOptions
from sklearn.metrics import confusion_matrix

def train(
        opt, 
        epoch, 
        dataloader_train, 
        summary_writer, 
        model, 
        optimizer, 
        train_criterion
    ):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('accuracy')
    model.train()
    num_iters = int(len(dataloader_train) / opt.batch_size)
    for i, (data, label) in enumerate(dataloader_train):
        data, label = data.to(opt.device), label.to(opt.device)
        optimizer.zero_grad()
        output = model(data)
        loss = train_criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss, data.shape[0])
        acc = accuracy(output, label)
        acc_meter.update(acc[0][0], data.shape[0])
        print(f'Train [{epoch}]/{opt.num_epoch}][{i}/{num_iters}]\t')
        print(f'{loss_meter}\t{acc_meter}')
    summary_writer.add_scalar('train_loss', loss_meter.avg, epoch)
    summary_writer.add_scalar('train_acc', acc_meter.avg, epoch)
    return loss_meter.avg
    

def val():
    loss = AverageMeter('loss')
def test(
        opt, 
        epoch, 
        dataloader_test, 
        summary_writer, 
        model, 
        test_criterion
    ):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('accuracy')
    model.eval()
    num_iters = int(len(dataloader_test) / opt.batch_size)
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader_test):
            data, label = data.to(opt.device), label.to(opt.device)
            output = model(data)
            loss = test_criterion(output, label)  # sum up batch loss
            loss_meter.update(loss, label.shape[0])
            acc = accuracy(output, label)
            acc_meter.update(acc[0][0], label.shape[0])
        print(f'Test [{epoch}]/{opt.num_epoch}][{i}/{num_iters}]\t')
        print(f'{loss_meter}\t{acc_meter}')
    summary_writer.add_scalar('test_loss', loss_meter.avg, epoch)
    summary_writer.add_scalar('test_acc', acc_meter.avg, epoch)
    return loss_meter.avg


def main():
    #parse arguments
    opt = TrainOptions().parse()
    opt.device = torch.device("cuda")

    #construct data loader
    dataloader_train = MNISTDataLoader(opt)
    dataset_size = len(dataloader_train)
    print('Dataset for train size is %d' % dataset_size)

    #create validation set data loader if validation_on option is set
    if opt.test_on:
        opt.mode = 'test'
        dataloader_test = MNISTDataLoader(opt)
        opt.mode = 'train'
        dataset_size = len(dataloader_test)
        print('Dataset for test size is %d' % dataset_size)

    # Tensorboard start
    summary_writer = SummaryWriter()

    # Network Builders
    model = MNISTModel(opt)
    # model = Net()
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-8)

    # Set up loss functions
    train_criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss()
    # train_criterion = nn.MSELoss()

    num_epoch = opt.num_epoch
    best_loss = float('inf')
    for epoch in range(num_epoch):
        start_time = time.time()
        train_loss = train(opt, epoch, dataloader_train, summary_writer, model, optimizer, train_criterion)
        if epoch >= 0:
            test_loss = test(opt, epoch, dataloader_test, summary_writer, model, test_criterion)
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