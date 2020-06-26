#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools

import numpy as np
import torch
from crypten.cuda import CUDALongTensor


class ConfigBase(abc.ABC):
    def __init__(self, config, *args):
        self.config = config
        assert len(args) % 2 == 0, "Uneven number of configuration params."
        self.params = args[::2]
        self.values = args[1::2]

    def __enter__(self):
        self.old_values = []
        for p, v in zip(self.params, self.values):
            self.old_values.append(getattr(self.config, p))
            setattr(self.config, p, v)

    def __exit__(self, exc_type, exc_value, tb):
        for p, v in zip(self.params, self.old_values):
            setattr(self.config, p, v)
        return exc_type is None


def count_wraps(share_list):
    """Computes the number of overflows or underflows in a set of shares

    We compute this by counting the number of overflows and underflows as we
    traverse the list of shares.
    """
    result = torch.zeros_like(share_list[0], dtype=torch.long)
    prev = share_list[0]
    for cur in share_list[1:]:
        next = cur + prev
        result -= ((prev < 0) & (cur < 0) & (next > 0)).long()  # underflow
        result += ((prev > 0) & (cur > 0) & (next < 0)).long()  # overflow
        prev = next
    return result


def pool_reshape(input, kernel_size, padding=None, stride=None, pad_value=0):
    """Rearrange a 4-d tensor so that each kernel is represented by each row"""
    # Setup kernel / stride values
    k = kernel_size
    if isinstance(k, int):
        k = (k, k)

    s = stride
    if s is None:
        s = k
    elif isinstance(s, int):
        s = (s, s)

    # Assert input parameters are correct type / size
    assert isinstance(k, tuple), "kernel_size must be an int or tuple"
    assert isinstance(s, tuple), "stride must be and int, a tuple, or None"
    assert len(k) == 2, "kernel_size must be an int or tuple pair"
    assert len(s) == 2, "stride must be an int or tuple pair"
    assert isinstance(pad_value, int), "pad_value must be an integer"
    assert input.dim() == 4, "pool input must be a 4-d tensor"

    # Apply padding if necessary
    if padding is not None:
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert len(padding) == 2, "Padding must be an integer or a pair"
        padding = (padding[0], padding[0], padding[1], padding[1])
        input = torch.nn.functional.pad(input, padding, value=pad_value)

    # Compute output size based on parameters
    n = input.size(0)
    c = input.size(1)
    h = (input.size(2) - k[0]) // s[0] + 1
    w = (input.size(3) - k[1]) // s[1] + 1
    out_size = (n, c, h, w)

    # Reshape input to arrange kernels to be represented by rows
    kernel_indices = torch.tensor(range(k[1]), device=input.device)
    kernel_indices = torch.cat(
        [kernel_indices + i * input.size(3) for i in range(k[0])]
    )
    kernel_indices = torch.stack([kernel_indices + i * s[0] for i in range(w)])

    offset = input.size(3)
    kernel_indices = torch.cat([kernel_indices + i * s[1] * offset for i in range(h)])

    offset *= input.size(2)
    kernel_indices = torch.stack(
        [kernel_indices + i * offset for i in range(input.size(1))]
    )

    offset *= input.size(1)
    kernel_indices = torch.stack(
        [kernel_indices + i * offset for i in range(input.size(0))]
    )

    input = input.take(kernel_indices)

    return input, out_size


@functools.lru_cache(maxsize=10)
def chebyshev_series(func, width, terms):
    r"""Computes Chebyshev coefficients

    For n = terms, the ith Chebyshev series coefficient is

    .. math::
        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))

    Args:
        func (function): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation

    Returns:
        Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = torch.arange(start=0, end=terms).float()
    x = width * torch.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = torch.cos(torch.ger(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * torch.sum(y * cos_term, axis=1)
    return coeffs


# FIXME: pytorch currently does not register `torch.cat` and
# `torch.stack` in __torch_function__. We therefore can not call
# torch.stack/torch.cat with CUDALongTensor as parameters. This is
# a temporary solution before pytorch fix their issue.
# See https://github.com/pytorch/pytorch/issues/34294 for details
def torch_cat(tensors, dim=0, out=None):
    is_cuda = any(t.is_cuda for t in tensors)
    if is_cuda:
        return CUDALongTensor.cat(tensors, dim=dim, out=out)
    return torch.cat(tensors, dim=dim, out=out)


def torch_stack(tensors, dim=0, out=None):
    is_cuda = any(t.is_cuda for t in tensors)
    if is_cuda:
        return CUDALongTensor.stack(tensors, dim=dim, out=out)
    return torch.stack(tensors, dim=dim, out=out)
