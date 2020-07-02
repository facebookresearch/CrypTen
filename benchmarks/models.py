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


def resnet18():
    model = torch.hub.load("pytorch/vision:v0.5.0", "resnet18", pretrained=True)
    return model


def resnet18_enc():
    model = torch.hub.load("pytorch/vision:v0.5.0", "resnet18", pretrained=True)
    dummy_input = torch.rand([1, 3, 224, 224])
    model_enc = crypten.nn.from_pytorch(model, dummy_input)
    return model_enc


MODELS = [
    Model(
        name="logistic regression",
        plain=LogisticRegression,
        crypten=LogisticRegressionCrypTen,
        data=data.GaussianClusters(),
        epochs=50,
        lr=0.1,
        loss="BCELoss",
        advanced=False,
    ),
    Model(
        name="feedforward neural network",
        plain=FeedForward,
        crypten=FeedForwardCrypTen,
        data=data.GaussianClusters(),
        epochs=50,
        lr=0.1,
        loss="BCELoss",
        advanced=False,
    ),
    Model(
        name="resnet 18",
        plain=resnet18,
        crypten=resnet18_enc,
        data=data.Images(),
        epochs=2,
        lr=0.1,
        loss="CrossEntropyLoss",
        advanced=True,
    ),
]
