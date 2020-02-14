#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains models used for benchmarking
"""


import crypten
import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LogisticRegressionCrypTen(crypten.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = crypten.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).sigmoid()


class FeedForward(torch.nn.Module):
    def __init__(self, n_features):
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
    def __init__(self, n_features):
        super().__init__()
        self.linear1 = crypten.nn.Linear(n_features, n_features // 2)
        self.linear2 = crypten.nn.Linear(n_features // 2, n_features // 4)
        self.linear3 = crypten.nn.Linear(n_features // 4, 1)

    def forward(self, x):
        out = (self.linear1(x)).relu()
        out = (self.linear2(out)).relu()
        out = (self.linear3(out)).sigmoid()
        return out
