#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..autograd_cryptensor import AutogradCrypTensor
from .module import Module


class _Loss(Module):
    """
    Base criterion class that mimics Pytorch's Loss.
    """

    def __init__(self, reduction="sum"):
        super(_Loss, self).__init__()
        if reduction != "sum":
            raise NotImplementedError("reduction %s not supported")
        self.reduction = reduction

    def forward(self, yhat, y):
        raise NotImplementedError("forward not implemented")

    def __call__(self, yhat, y):
        return self.forward(yhat, y)


class MSELoss(_Loss):
    """
    Mean squared error between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        N = y.nelement()
        assert (
            yhat.nelement() == N
        ), "input and target must have the same number of elements"
        return (yhat - y).square().sum().div(N)


class L1Loss(_Loss):
    """
    Mean absolute error between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        N = y.nelement()
        assert (
            yhat.nelement() == N
        ), "input and target must have the same number of elements"
        return (yhat - y).abs().sum().div(N)


class BCELoss(_Loss):
    """
    Binary cross-entropy loss between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        assert (
            yhat.nelement() == y.nelement()
        ), "input and target must have the same number of elements"
        assert all(
            isinstance(val, AutogradCrypTensor) for val in [y, yhat]
        ), "inputs must be AutogradCrypTensors"
        return yhat.binary_cross_entropy(y)


class CrossEntropyLoss(_Loss):
    """
    Cross-entropy loss between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        assert yhat.size() == y.size(), "input and target must have the same size"
        assert all(
            isinstance(val, AutogradCrypTensor) for val in [y, yhat]
        ), "inputs must be AutogradCrypTensors"
        return yhat.cross_entropy(y)
