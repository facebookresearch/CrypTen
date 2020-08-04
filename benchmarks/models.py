#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains models used for benchmarking
"""


from dataclasses import dataclass
from typing import Any

import crypten
import torch
import torch.nn as nn
from torchvision import models


try:
    from . import data
except ImportError:
    # direct import if relative fails
    import data


N_FEATURES = 20


@dataclass
class Model:
    name: str
    plain: torch.nn.Module
    crypten: crypten.nn.Module
    # must contains x, y, x_test, y_test attributes
    data: Any
    epochs: int
    lr: float
    loss: str
    advanced: bool


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LogisticRegressionCrypTen(crypten.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear = crypten.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).sigmoid()


class FeedForward(torch.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, n_features // 2)
        self.linear2 = torch.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = torch.nn.Linear(n_features // 4, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.sigmoid(self.linear3(out))
        return out


class FeedForwardCrypTen(crypten.nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear1 = crypten.nn.Linear(n_features, n_features // 2)
        self.linear2 = crypten.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = crypten.nn.Linear(n_features // 4, 1)

    def forward(self, x):
        out = (self.linear1(x)).relu()
        out = (self.linear2(out)).relu()
        out = (self.linear3(out)).sigmoid()
        return out


class ResNet(nn.Module):
    def __init__(self, n_layers=18):
        super().__init__()
        assert n_layers in [18, 34, 50]
        self.model = getattr(models, "resnet{}".format(n_layers))(pretrained=True)

    def forward(self, x):
        return self.model(x)


class ResNetCrypTen(crypten.nn.Module):
    def __init__(self, n_layers=18):
        super().__init__()
        assert n_layers in [18, 34, 50]
        model = getattr(models, "resnet{}".format(n_layers))(pretrained=True)
        dummy_input = torch.rand([1, 3, 224, 224])
        self.model = crypten.nn.from_pytorch(model, dummy_input)

    def forward(self, x):
        return self.model(x)


MODELS = [
    Model(
        name="logistic regression",
        plain=LogisticRegression(),
        crypten=LogisticRegressionCrypTen(),
        data=data.GaussianClusters(),
        epochs=50,
        lr=0.1,
        loss="BCELoss",
        advanced=False,
    ),
    Model(
        name="feedforward neural network",
        plain=FeedForward(),
        crypten=FeedForwardCrypTen(),
        data=data.GaussianClusters(),
        epochs=50,
        lr=0.1,
        loss="BCELoss",
        advanced=False,
    ),
    Model(
        name="resnet18",
        plain=ResNet(n_layers=18),
        crypten=ResNetCrypTen(n_layers=18),
        data=data.Images(),
        epochs=2,
        lr=0.1,
        loss="CrossEntropyLoss",
        advanced=True,
    ),
    Model(
        name="resnet34",
        plain=ResNet(n_layers=34),
        crypten=ResNetCrypTen(n_layers=34),
        data=data.Images(),
        epochs=2,
        lr=0.1,
        loss="CrossEntropyLoss",
        advanced=True,
    ),
    Model(
        name="resnet50",
        plain=ResNet(n_layers=50),
        crypten=ResNetCrypTen(n_layers=50),
        data=data.Images(),
        epochs=2,
        lr=0.1,
        loss="CrossEntropyLoss",
        advanced=True,
    ),
]
