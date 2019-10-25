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

    def __init__(self, reduction="mean"):
        super(_Loss, self).__init__()
        if reduction != "mean":
            raise NotImplementedError("reduction %s not supported")
        self.reduction = reduction

    def forward(self, x, y):
        raise NotImplementedError("forward not implemented")

    def __call__(self, x, y):
        return self.forward(x, y)


class MSELoss(_Loss):
    r"""
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = (x_n - y_n)^2,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.

    """

    def forward(self, x, y):
        N = y.nelement()
        assert (
            x.nelement() == N
        ), "input and target must have the same number of elements"
        return (x - y).square().sum().div(N)


class L1Loss(_Loss):
    r"""
    Creates a criterion that measures the mean absolute error between each element in
    the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = \left | x_n - y_n \right |,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.
    """

    def forward(self, x, y):
        N = y.nelement()
        assert (
            x.nelement() == N
        ), "input and target must have the same number of elements"
        return (x - y).abs().sum().div(N)


class BCELoss(_Loss):
    r"""
    Creates a criterion that measures the Binary Cross Entropy
    between the prediction :math:`x` and the target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right ],

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.
    """

    def forward(self, x, y):
        assert (
            x.nelement() == y.nelement()
        ), "input and target must have the same number of elements"
        assert all(
            isinstance(val, AutogradCrypTensor) for val in [y, x]
        ), "inputs must be AutogradCrypTensors"
        return x.binary_cross_entropy(y)


class CrossEntropyLoss(_Loss):
    r"""
    Creates a criterion that measures cross-entropy loss between the
    prediction :math:`x` and the target :math:`y`. It is useful when
    training a classification problem with `C` classes.

    The prediction `x` is expected to contain raw, unnormalized scores for each class.

    The prediction `x` has to be a Tensor of size either :math:`(N, C)` or
    :math:`(N, C, d_1, d_2, ..., d_K)`, where :math:`N` is the size of the minibatch,
    and with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    target `y` for each value of a 1D tensor of size `N`.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log \left(
        \frac{\exp(x[class])}{\sum_j \exp(x[j])} \right )
        = -x[class] + \log \left (\sum_j \exp(x[j]) \right)

    The losses are averaged across observations for each batch

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape.
    """

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        assert all(
            isinstance(val, AutogradCrypTensor) for val in [y, x]
        ), "inputs must be AutogradCrypTensors"
        return x.cross_entropy(y)
