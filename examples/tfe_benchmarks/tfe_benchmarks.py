#!/usr/bin/env python3
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
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter
from torchvision import datasets, transforms


def run_tfe_benchmarks(
    network="B",
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
    def preprocess_data():
        mnist_train = datasets.MNIST(
            "/tmp",
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        mnist_test = datasets.MNIST(
            "/tmp",
            download=True,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        train_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    train_loader, val_loader = preprocess_data()
    flatten = False
    if network == "A":
        flatten = True

    if evaluate:
        if not skip_plaintext:
            logging.info("===== Evaluating plaintext benchmark network =====")
            validate(val_loader, model, criterion, print_freq, flatten=flatten)
        private_model = create_private_benchmark_model(model)
        logging.info("===== Evaluating Private benchmark network =====")
        validate(val_loader, private_model, criterion, print_freq, flatten=flatten)
        # validate_side_by_side(val_loader, model, private_model, flatten=flatten)
        return

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
        checkpoint_file = "../examples/benchmarks/checkpoint_bn" + network + ".pth.tar"
        model_best_file = "../examples/benchmarks/model_best_bn" + network + ".pth.tar"
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "Benchmark" + network,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=checkpoint_file,
            model_best=model_best_file,
        )


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
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )


def validate_side_by_side(val_loader, model0, model1, flatten=False):
    # switch to evaluate mode
    model0.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            if flatten:
                input = input.view(input.size(0), -1)
            output0 = model0(input)
            output1 = model1(input)
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output0)
            logging.info("Encrypted:\n%s\n" % output1)
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
            if isinstance(model, crypten.nn.Module) and not isinstance(
                input, crypten.MPCTensor
            ):
                input = crypten.MPCTensor(input)

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
            if i > 100:
                break

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )

    return top1.value()


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", model_best="model_best.pth.tar"
):
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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


def create_private_benchmark_model(model):
    dummy_input = torch.rand((1, 1, 28, 28))
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
        out = out.view(out.size(0), -1)
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
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.batchnorm3(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
