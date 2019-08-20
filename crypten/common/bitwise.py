#!/usr/bin/env python3

# dependencies:
import numpy as np
import torch


def get_bit(tensor, index):
    """
    Returns a tensor with 1's for elements where the `index`th bit is 1, and 0's
    otherwise.
    """
    return (tensor >> index) & 1


def set_bit(tensor, index, value):
    """
    Returns a tensor with the `index`th bit of the input set to `val`.
    """
    if index < 63:
        mask = 1 << index
    elif index == 63:
        mask = -(2 ** 63)
    else:
        mask = 0

    if torch.is_tensor(value):
        one = tensor | mask
        zero = tensor & ~mask
        return one * value + zero * (1 - value)

    if value == 0:
        return tensor & (~mask)
    else:
        return tensor | mask


def invert(tensor):
    """Invert each element in the tensor bitwise."""
    return torch.from_numpy(np.invert(tensor.numpy()))
