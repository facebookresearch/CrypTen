#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import shutil
import tempfile
import time
import warnings

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter
from examples.util import NoopContextManager, process_mnist_files
from torchvision import datasets, transforms


def run_tfe_benchmarks(
    network="B",
    epochs=5,
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
    save_checkpoint_dir="/tmp/tfe_benchmarks",
    save_modelbest_dir="/tmp/tfe_benchmarks_best",
    context_manager=None,
    mnist_dir=None,
):
    crypten.init()

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # create model
    model = create_benchmark_model(network)

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

    # Loading MNIST. Normalizing per pytorch/examples/blob/master/mnist/main.py
    def preprocess_data(context_manager, data_dirname):
        if mnist_dir is not None:
            process_mnist_files(
                mnist_dir, os.path.join(data_dirname, "MNIST", "processed")
            )
            download = False
        else:
            download = True

        with context_manager:
            if not evaluate:
                mnist_train = datasets.MNIST(
                    data_dirname,
                    download=download,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                )

            mnist_test = datasets.MNIST(
                data_dirname,
                download=download,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
        train_loader = (
            torch.utils.data.DataLoader(
                mnist_train, batch_size=batch_size, shuffle=True
            )
            if not evaluate
            else None
        )
        test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    if context_manager is None:
        context_manager = NoopContextManager()

    warnings.filterwarnings("ignore")
    data_dir = tempfile.TemporaryDirectory()
    train_loader, val_loader = preprocess_data(context_manager, data_dir.name)

    flatten = False
    if network == "A":
        flatten = True

    if evaluate:
        if not skip_plaintext:
            logging.info("===== Evaluating plaintext benchmark network =====")
            validate(val_loader, model, criterion, print_freq, flatten=flatten)
        private_model = create_private_benchmark_model(model, flatten=flatten)
        logging.info("===== Evaluating Private benchmark network =====")
        validate(val_loader, private_model, criterion, print_freq, flatten=flatten)
        # validate_side_by_side(val_loader, model, private_model, flatten=flatten)
        return

    os.makedirs(save_checkpoint_dir, exist_ok=True)
    os.makedirs(save_modelbest_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            print_freq,
            flatten=flatten,
        )

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, print_freq, flatten=flatten)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_file = "checkpoint_bn" + network + ".pth.tar"
        model_best_file = "model_best_bn" + network + ".pth.tar"
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "Benchmark" + network,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=os.path.join(save_checkpoint_dir, checkpoint_file),
            model_best=os.path.join(save_modelbest_dir, model_best_file),
        )
    data_dir.cleanup()
    shutil.rmtree(save_checkpoint_dir)


def train(
    train_loader, model, criterion, optimizer, epoch, print_freq=10, flatten=False
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # compute output
        if flatten:
            input = input.view(input.size(0), -1)
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


def validate_side_by_side(val_loader, plaintext_model, private_model, flatten=False):
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            if flatten:
                input = input.view(input.size(0), -1)
            output0 = plaintext_model(input)
            encr_input = crypten.cryptensor(input)
            output1 = private_model(encr_input)
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output0)
            logging.info("Encrypted:\n%s\n" % output1.get_plain_text())
            if i > 1000:
                break


def validate(val_loader, model, criterion, print_freq=10, flatten=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            if flatten:
                input = input.view(input.size(0), -1)
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = crypten.cryptensor(input)

            output = model(input)

            if crypten.is_encrypted_tensor(output):
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
            if i > 100:
                break

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )

    return top1.value()


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", model_best="model_best.pth.tar"
):
    # TODO: use crypten.save_from_party() in future.
    rank = comm.get().get_rank()
    # only save for process rank = 0
    if rank == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, model_best)


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
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_benchmark_model(benchmark):
    if benchmark == "A":
        return NetworkA()
    elif benchmark == "B":
        return NetworkB()
    elif benchmark == "C":
        return NetworkC()
    else:
        raise RuntimeError("Invalid benchmark network")


def create_private_benchmark_model(model, flatten=False):
    dummy_input = torch.empty((1, 1, 28, 28))
    if flatten:
        dummy_input = torch.empty((1, 28 * 28))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt()
    return private_model


class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)

    def forward(self, x):
        out = self.fc1(x)
        out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class NetworkB(nn.Module):
    def __init__(self):
        super(NetworkB, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 16 * 4 * 4)
        out = self.fc1(out)
        out = self.batchnorm3(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class NetworkC(nn.Module):
    def __init__(self):
        super(NetworkC, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.batchnorm1 = nn.BatchNorm2d(20)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.batchnorm3 = nn.BatchNorm1d(500)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 50 * 4 * 4)
        out = self.fc1(out)
        out = self.batchnorm3(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
