#!/usr/bin/env python3

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
        return (yhat - y).square().sum().div_(N)


class L1Loss(_Loss):
    """
    Mean absolute error between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        N = y.nelement()
        assert (
            yhat.nelement() == N
        ), "input and target must have the same number of elements"
        return (yhat - y).abs().sum().div_(N)


class BCELoss(_Loss):
    """
    Binary cross-entropy loss between predictions and ground-truth values.
    """

    def forward(self, yhat, y):
        N = y.nelement()
        assert (
            yhat.nelement() == N
        ), "input and target must have the same number of elements"
        retval = y * yhat.log() + ((1 - y) * (1 - yhat).log())
        return -retval.sum().div_(N)

    # def backward(self, p, y):
    #     retval = ((1 - p).reciprocal() * (1 - y)) - p.reciprocal() * y
    #     return retval.div_(y.nelement())
