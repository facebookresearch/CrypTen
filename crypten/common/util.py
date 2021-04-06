#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools
import math

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
        print(self.params)
        print(self.values)
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


def _pooling_output_shape(
    input_size, kernel_size, pad_l, pad_r, stride, dilation, ceil_mode
):
    """
    Generates output shape along a single dimension following conventions here:
    https://github.com/pytorch/pytorch/blob/b0424a895c878cb865947164cb0ce9ce3c2e73ef/aten/src/ATen/native/Pool.h#L24-L38
    """
    numerator = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        numerator += stride - 1

    output_size = numerator // stride + 1

    # ensure that the last pooling starts inside the image
    # needed to avoid problems in ceil mode
    if ceil_mode and (output_size - 1) * stride >= input_size + pad_l:
        output_size -= 1

    return output_size


def pool2d_reshape(
    input,
    kernel_size,
    padding=None,
    stride=None,
    dilation=1,
    ceil_mode=False,
    pad_value=0,
):
    """Rearrange a 4-d tensor so that each kernel is represented by each row"""

    # Setup kernel / stride / dilation values
    k = kernel_size
    if isinstance(k, int):
        k = (k, k)

    s = stride
    if s is None:
        s = k
    elif isinstance(s, int):
        s = (s, s)

    d = dilation
    if isinstance(d, int):
        d = (d, d)

    # Assert input parameters are correct type / size
    assert isinstance(k, tuple), "kernel_size must be an int or tuple"
    assert isinstance(s, tuple), "stride must be and int, a tuple, or None"
    assert len(k) == 2, "kernel_size must be an int or tuple pair"
    assert len(s) == 2, "stride must be an int or tuple pair"
    assert isinstance(pad_value, int), "pad_value must be an integer"
    assert input.dim() >= 2, "Pooling input dimension should be at least 2"

    # Apply padding if necessary
    if padding is not None:
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert len(padding) == 2, "Padding must be an integer or a pair"
        padding = (padding[0], padding[0], padding[1], padding[1])
    else:
        padding = (0, 0, 0, 0)

    # Compute output size based on parameters
    n = input.size()[:-2]
    h = _pooling_output_shape(
        input.size(-2), k[0], padding[0], padding[1], s[0], d[0], ceil_mode
    )
    w = _pooling_output_shape(
        input.size(-1), k[1], padding[2], padding[3], s[1], d[1], ceil_mode
    )

    out_size = tuple(n + (h, w))

    input = torch.nn.functional.pad(input, padding, value=pad_value)
    if ceil_mode:
        update_pad = [0, 0, 0, 0]
        update_pad[3] = h * s[0] + (k[0] - 1) * d[0] - input.size(-2)
        update_pad[1] = w * s[1] + (k[1] - 1) * d[1] - input.size(-1)
        input = torch.nn.functional.pad(input, tuple(update_pad), value=pad_value)

    # Reshape input to arrange kernels to be represented by rows
    kernel_indices = torch.tensor(range(0, k[1] * d[1], d[1]), device=input.device)
    kernel_indices = torch.cat(
        [kernel_indices + i * input.size(-1) for i in range(0, k[0] * d[0], d[0])]
    )
    kernel_indices = torch.stack([kernel_indices + i * s[1] for i in range(w)])

    offset = input.size(-1)
    kernel_indices = torch.cat([kernel_indices + i * s[0] * offset for i in range(h)])

    for dim in range(2, input.dim()):
        offset *= input.size(-dim)
        kernel_indices = torch.stack(
            [kernel_indices + i * offset for i in range(input.size(-dim - 1))]
        )

    output = input.take(kernel_indices)
    return output, out_size


def adaptive_pool2d_helper(input, output_size, reduction="mean"):
    r"""
    Provides a helper that adapts the input size and provides input
    args / kwargs to allow pool2d functions to emulate adaptive pool2d
    functions.

    This function computes the kernel_size, stride, and padding for
    pool2d functions and inserts rows along each dimension so that
    a constant stride can be used.
    """
    import crypten

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    assert len(output_size) == 2, "output_size must be 2-dimensional."

    output_size = list(output_size)
    for i in range(2):
        if output_size[i] is None:
            output_size[i] = input.size(i - 2)

    # Compute the start_index and end_index for kernels
    def compute_kernels(in_size, out_size):
        step = in_size / out_size

        starts = []
        ends = []
        max_kernel_size = 0
        for j in range(out_size):
            # Compute local kernel size
            start_index = int(j * step)
            end_index = int(math.ceil((j + 1) * step))
            k = end_index - start_index

            # Update global kernel size
            max_kernel_size = k if k > max_kernel_size else max_kernel_size

            # Store local kernels
            starts.append(start_index)
            ends.append(end_index)

        return starts, ends, max_kernel_size

    # Repeats a row `ind` of `tensor` at dimension `dim` for overlapping kernels
    def repeat_row(tensor, dim, ind):
        x = tensor.index_select(dim, torch.arange(ind))
        y = tensor.index_select(dim, torch.arange(ind, tensor.size(dim)))
        repeated_row = tensor.index_select(dim, torch.tensor(ind - 1))
        return crypten.cat([x, repeated_row, y], dim=dim)

    # Extends a row where a kernel is smaller than the maximum kernel size
    def extend_row(tensor, dim, start_ind, end_ind):
        if reduction == "mean":
            extended_value = tensor.index_select(dim, torch.arange(start_ind, end_ind))
            extended_value = extended_value.mean(dim, keepdim=True)
        elif reduction == "max":
            extended_value = tensor.index_select(dim, torch.tensor(start_ind))
        else:
            raise ValueError(f"Invalid reduction {reduction} for adaptive pooling.")

        if start_ind == 0:
            return crypten.cat([extended_value, tensor], dim=dim)

        x = tensor.index_select(dim, torch.arange(start_ind))
        y = tensor.index_select(dim, torch.arange(start_ind, tensor.size(dim)))
        return crypten.cat([x, extended_value, y], dim=dim)

    strides = []
    for i in range(2):
        dim = i - 2 + input.dim()
        in_size = input.size(dim)
        out_size = output_size[i] if output_size[i] is not None else in_size

        # Compute repeats
        if out_size > 1:
            starts, ends, stride = compute_kernels(in_size, out_size)

            added_rows = 0
            for i in range(out_size):
                start_ind = starts[i]
                end_ind = ends[i]

                # Extend kernel so all kernels have the same size
                k = end_ind - start_ind
                for _ in range(k, stride):
                    input = extend_row(
                        input, dim, start_ind + added_rows, end_ind + added_rows
                    )
                    added_rows += 1

                if i == out_size - 1:
                    break

                # Repeat overlapping rows so stride can be equal to the kernel size
                if end_ind > starts[i + 1]:
                    input = repeat_row(input, dim, end_ind + added_rows)
                    added_rows += 1
        else:
            stride = in_size

        strides.append(stride)

    strides = tuple(strides)
    kernel_sizes = strides

    args = (kernel_sizes,)
    kwargs = {"stride": strides}

    return input, args, kwargs


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
