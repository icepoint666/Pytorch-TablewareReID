import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import time

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets.data_manager import Tableware
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models import get_baseline_model
from evaluator import Evaluator
from utils.meters import AverageMeter
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform

def triplet_example(input1, labels):
    labels_set = set(labels.numpy())
    label_to_indices = {label: np.where(labels.numpy() == label)[0] for label in labels_set}
    random_state = np.random.RandomState(29)
    input2 = torch.Tensor(input1.size())
    input3 = torch.Tensor(input1.size())

    for i, label in enumerate(labels.numpy()):
        p_idx = i
        while p_idx == i:
            if len(label_to_indices[label]) == 1:
                break
            p_idx = random_state.choice(label_to_indices[label])
        input2[i] = input1[p_idx]
        n_idx = random_state.choice(label_to_indices[
                random_state.choice(list(labels_set - set([label])))])
        input3[i] = input1[n_idx]
    return input1, input2, input3

def train(model, optimizer, criterion, epoch, print_freq, data_loader):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - start)

        # model optimizer
        # parse data
        imgs, pids = inputs

        img1, img2, img3 = triplet_example(imgs, pids)
        input1 = img1.cuda()
        input2 = img2.cuda()
        input3 = img3.cuda()

        # forward
        feat1 = model(input1)
        feat2 = model(input2)
        feat3 = model(input3)

        loss = criterion(feat1, feat2, feat3)

        optimizer.zero_grad()
        # backward
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        losses.update(loss.item())

        start = time.time()

        if (i + 1) % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Batch Time {:.3f} ({:.3f})\t'
                    'Data Time {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
    print()

def trainer():
    seed = 0

    # dataset options
    height = 128
    width = 128

    # optimization options
    optim = 'Adam'
    max_epoch = 1
    train_batch = 8
    test_batch = 32
    lr = 0.1
    step_size = 40
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    margin = 0.3
    num_instances = 4
    num_gpu = 1

    # model options
    last_stride = 1
    pretrained_model = 'model/resnet50-19c8e357.pth'

    # miscs
    print_freq = 2
    eval_step = 50
    save_dir = 'model/pytorch-ckpt/'
    workers = 1
    start_epoch = 0

    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print('currently using cpu')

    pin_memory = True if use_gpu else False

    print('initializing dataset {}'.format('Tableware'))
    dataset = Tableware('/home/icepoint/reid_tableware/datas/transdatas/')

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(height, width)),

        batch_size=train_batch, num_workers=workers,
        pin_memory=pin_memory, drop_last=True
    )

    testloader = DataLoader(
        ImageData(dataset.test, TestTransform(height, width)),
        batch_size=test_batch, num_workers=workers,
        pin_memory=pin_memory, drop_last=True
    )

    model, optim_policy = get_baseline_model(model_path=pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    tri_criterion = TripletLoss(margin)

    # get optimizer
    optimizer = torch.optim.Adam(
        optim_policy, lr=lr, weight_decay=weight_decay
    )

    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * num_gpu
        elif ep < 180:
            lr = 1e-4 * num_gpu
        elif ep < 300:
            lr = 1e-5 * num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) *num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * num_gpu
        else:
            lr = 1e-5 * num_gpu
        for p in optimizer.param_groups:
            p['lr'] = lr

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    evaluator = Evaluator(model)
    # start training
    best_acc = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, max_epoch):
        if step_size > 0:
            adjust_lr(optimizer, epoch + 1)

        train(model, optimizer, tri_criterion, epoch, print_freq, trainloader)

        # skip if not save model
        if eval_step > 0 and (epoch + 1) % eval_step == 0 or (epoch + 1) == max_epoch:
            acc = evaluator.evaluate(testloader, 10.0)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=is_best, save_dir=save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print(
        'Best accuracy {:.1%}, achieved at epoch {}'.format(best_acc, best_epoch))

if __name__ == "__main__":
    trainer()
