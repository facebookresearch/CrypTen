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
