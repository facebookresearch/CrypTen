#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import crypten
import torch
from examples.meters import AverageMeter


def train_linear_svm(features, labels, epochs=50, lr=0.5, print_time=False):
    # Initialize random weights
    w = features.new(torch.randn(1, features.size(0)))
    b = features.new(torch.randn(1))

    if print_time:
        pt_time = AverageMeter()
        end = time.time()

    for epoch in range(epochs):
        # Forward
        label_predictions = w.matmul(features).add(b).sign()

        # Compute accuracy
        correct = label_predictions.mul(labels)
        accuracy = correct.add(1).div(2).mean()
        if crypten.is_encrypted_tensor(accuracy):
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        if crypten.communicator.get().get_rank() == 0:
            print(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
            )

        # Backward
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
        b_grad = loss_grad.mean()
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1))

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            end = time.time()

    return w, b


def evaluate_linear_svm(features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))


def run_mpc_linear_svm(
    epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False
):
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Initialize x, y, w, b
    x = torch.randn(features, examples)
    w_true = torch.randn(1, features)
    b_true = torch.randn(1)
    y = w_true.matmul(x) + b_true
    y = y.sign()

    if not skip_plaintext:
        logging.info("==================")
        logging.info("PyTorch Training")
        logging.info("==================")
        w_torch, b_torch = train_linear_svm(x, y, lr=lr, print_time=True)

    # Encrypt features / labels
    x = crypten.cryptensor(x)
    y = crypten.cryptensor(y)

    logging.info("==================")
    logging.info("CrypTen Training")
    logging.info("==================")
    w, b = train_linear_svm(x, y, lr=lr, print_time=True)

    if not skip_plaintext:
        logging.info("PyTorch Weights  :")
        logging.info(w_torch)
    logging.info("CrypTen Weights:")
    logging.info(w.get_plain_text())

    if not skip_plaintext:
        logging.info("PyTorch Bias  :")
        logging.info(b_torch)
    logging.info("CrypTen Bias:")
    logging.info(b.get_plain_text())
