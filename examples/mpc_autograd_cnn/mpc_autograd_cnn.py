#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.util import NoopContextManager
from torchvision import datasets, transforms


def run_mpc_autograd_cnn(
    context_manager=None,
    num_epochs=3,
    learning_rate=0.001,
    batch_size=5,
    print_freq=5,
    num_samples=100,
):
    """
    Args:
        context_manager: used for setting proxy settings during download.
    """
    crypten.init()

    data_alice, data_bob, train_labels = preprocess_mnist(context_manager)
    rank = comm.get().get_rank()

    # assumes at least two parties exist
    # broadcast dummy data with same shape to remaining parties
    if rank == 0:
        x_alice = data_alice
    else:
        x_alice = torch.empty(data_alice.size())

    if rank == 1:
        x_bob = data_bob
    else:
        x_bob = torch.empty(data_bob.size())

    # encrypt
    x_alice_enc = crypten.cryptensor(x_alice, src=0)
    x_bob_enc = crypten.cryptensor(x_bob, src=1)

    # combine feature sets
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    x_combined_enc = x_combined_enc.unsqueeze(1)

    # reduce training set to num_samples
    x_reduced = x_combined_enc[:num_samples]
    y_reduced = train_labels[:num_samples]

    # encrypt plaintext model
    model_plaintext = CNN()
    dummy_input = torch.empty((1, 1, 28, 28))
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
    model.train()
    model.encrypt()

    # encrypted training
    train_encrypted(
        x_reduced, y_reduced, model, num_epochs, learning_rate, batch_size, print_freq
    )


def train_encrypted(
    x_encrypted,
    y_encrypted,
    encrypted_model,
    num_epochs,
    learning_rate,
    batch_size,
    print_freq,
):
    rank = comm.get().get_rank()
    loss = crypten.nn.MSELoss()

    num_samples = x_encrypted.size(0)
    label_eye = torch.eye(2)

    for epoch in range(num_epochs):
        last_progress_logged = 0
        # only print from rank 0 to avoid duplicates for readability
        if rank == 0:
            print(f"Epoch {epoch} in progress:")

        for j in range(0, num_samples, batch_size):

            # define the start and end of the training mini-batch
            start, end = j, min(j + batch_size, num_samples)

            # switch on autograd for training examples
            x_train = x_encrypted[start:end]
            x_train.requires_grad = True
            y_one_hot = label_eye[y_encrypted[start:end]]
            y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            # perform forward pass:
            output = encrypted_model(x_train)
            loss_value = loss(output, y_train)

            # backprop
            encrypted_model.zero_grad()
            loss_value.backward()
            encrypted_model.update_parameters(learning_rate)

            # log progress
            if j + batch_size - last_progress_logged >= print_freq:
                last_progress_logged += print_freq
                print(f"Loss {loss_value.get_plain_text().item():.4f}")

        # compute accuracy every epoch
        pred = output.get_plain_text().argmax(1)
        correct = pred.eq(y_encrypted[start:end])
        correct_count = correct.sum(0, keepdim=True).float()
        accuracy = correct_count.mul_(100.0 / output.size(0))

        loss_plaintext = loss_value.get_plain_text().item()
        print(
            f"Epoch {epoch} completed: "
            f"Loss {loss_plaintext:.4f} Accuracy {accuracy.item():.2f}"
        )


def preprocess_mnist(context_manager):
    if context_manager is None:
        context_manager = NoopContextManager()

    with context_manager:
        # each party gets a unique temp directory
        with tempfile.TemporaryDirectory() as data_dir:
            mnist_train = datasets.MNIST(data_dir, download=True, train=True)
            mnist_test = datasets.MNIST(data_dir, download=True, train=False)

    # modify labels so all non-zero digits have class label 1
    mnist_train.targets[mnist_train.targets != 0] = 1
    mnist_test.targets[mnist_test.targets != 0] = 1
    mnist_train.targets[mnist_train.targets == 0] = 0
    mnist_test.targets[mnist_test.targets == 0] = 0

    # compute normalization factors
    data_all = torch.cat([mnist_train.data, mnist_test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0)

    # normalize data
    data_train_norm = transforms.functional.normalize(
        mnist_train.data.float(), tensor_mean, tensor_std
    )

    # partition features between Alice and Bob
    data_alice = data_train_norm[:, :, :20]
    data_bob = data_train_norm[:, :, 20:]
    train_labels = mnist_train.targets

    return data_alice, data_bob, train_labels


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
