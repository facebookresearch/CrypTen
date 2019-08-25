#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import shutil
import time
import warnings

import crypten
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter


def run_mpc_cifar(
    data,
    epochs=25,
    start_epoch=0,
    batch_size=256,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-6,
    print_freq=10,
    resume="",
    evaluate=True,
    seed=None,
    skip_plaintext=False,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # create model
    model = LeNet()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )

    # optionally resume from a checkpoint
    best_prec1 = 0
    if resume:
        if os.path.isfile(resume):
            logging.info("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(resume))

    # Data loading code
    def preprocess_data(file):
        def unpickle(file):
            import pickle

            with open(file, "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
            return dict

        raw_data = unpickle(file)
        labels = raw_data[b"labels"]
        loader = []
        for i in range(len(labels) // batch_size):
            inds = slice(i * batch_size, (i + 1) * batch_size)
            loader.append(
                (
                    torch.from_numpy(
                        raw_data[b"data"][inds].reshape((batch_size, 3, 32, 32))
                    ).to(dtype=torch.float32),
                    torch.tensor(labels[inds]),
                )
            )
        return loader

    valdir = os.path.join(data, "test_batch")
    val_loader = preprocess_data(valdir)

    if evaluate:
        if not skip_plaintext:
            logging.info("===== Evaluating plaintext LeNet network =====")
            validate(val_loader, model, criterion, print_freq)
        dummy_input = torch.rand((1, 3, 32, 32))
        private_model = crypten.nn.from_pytorch(model, dummy_input).encrypt()
        logging.info("===== Evaluating Private LeNet network =====")
        validate(val_loader, private_model, criterion, print_freq)
        return

    train_loader = []
    for i in range(1, 6):
        train_dir_i = os.path.join(data, "data_batch_%d" % i)
        train_loader_i = preprocess_data(train_dir_i)
        train_loader.extend(train_loader_i)

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "LeNet",
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.add(loss.item(), input.size(0))
        top1.add(prec1[0], input.size(0))
        top5.add(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        current_batch_time = time.time() - end
        batch_time.add(current_batch_time)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                "Epoch: [{}][{}/{}]\t"
                "Time {:.3f} ({:.3f})\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 {:.3f} ({:.3f})\t"
                "Prec@5 {:.3f} ({:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    current_batch_time,
                    batch_time.value(),
                    loss.item(),
                    losses.value(),
                    prec1[0],
                    top1.value(),
                    prec5[0],
                    top5.value(),
                )
            )


def validate_side_by_side(val_loader, model0, model1):
    # switch to evaluate mode
    model0.eval()
    val_loader = val_loader[:1000]

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output0 = model0(input)
            output1 = model1(input)
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output0)
            logging.info("Encrypted:\n%s\n" % output1)


def validate(val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    val_loader = val_loader[:100]

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not isinstance(
                input, crypten.MPCTensor
            ):
                input = crypten.MPCTensor(input)
            # compute output
            output = model(input)
            if isinstance(output, crypten.MPCTensor):
                output = output.get_plain_text()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top5.add(prec5[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )

    return top1.value()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations.
    """

    def __init__(self):
        super(LeNet, self).__init__()

        # network architecture:
        modules = [
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            Reshape(-1, 16 * 5 * 5),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        ]

        # register all modules:
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
