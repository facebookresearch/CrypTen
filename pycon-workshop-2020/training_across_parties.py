#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import crypten
import torch
import torchvision
from multiprocess_launcher import MultiProcessLauncher


# python 3.7 is required
assert sys.version_info[0] == 3 and sys.version_info[1] == 7, "python 3.7 is required"


# Alice is party 0
ALICE = 0
BOB = 1


class LogisticRegression(crypten.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = crypten.nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.linear(x)


def encrypt_digits():
    """Alice has images. Bob has labels"""
    digits = torchvision.datasets.MNIST(
        root="/tmp/data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    images, labels = take_samples(digits)
    crypten.save_from_party(images, "/tmp/data/alice_images.pth", src=ALICE)
    crypten.save_from_party(labels, "/tmp/data/bob_labels.pth", src=BOB)


def take_samples(digits, n_samples=100):
    """Returns images and labels based on sample size"""
    images, labels = [], []

    for i, digit in enumerate(digits):
        if i == n_samples:
            break
        image, label = digit
        images.append(image)
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), 10)
        labels.append(label_one_hot)

    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, labels


def train_model(model, X, y, epochs=10, learning_rate=0.05):
    criterion = crypten.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        print(f"epoch {epoch} loss: {loss.get_plain_text()}")
        loss.backward()
        model.update_parameters(learning_rate)
    return model


def jointly_train():
    encrypt_digits()
    alice_images_enc = crypten.load_from_party("/tmp/data/alice_images.pth", src=ALICE)
    bob_labels_enc = crypten.load_from_party("/tmp/data/bob_labels.pth", src=BOB)

    model = LogisticRegression().encrypt()
    model = train_model(model, alice_images_enc, bob_labels_enc)


if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, jointly_train)
    launcher.start()
    launcher.join()
    launcher.terminate()
