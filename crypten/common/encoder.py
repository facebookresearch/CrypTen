#!/usr/bin/env python3
import numpy as np
import torch

from .encrypted_tensor import EncryptedTensor
from .tensor_types import is_float_tensor, is_int_tensor


def nearest_integer_division(tensor, integer):
    """Performs division of integer tensor, rounding to nearest integer."""
    assert integer > 0, "only supports positive divisors"
    assert is_int_tensor(tensor), "unsupported type: %s" % type(tensor)

    lez = (tensor < 0).long()
    pos_remainder = (1 - lez) * tensor % integer
    neg_remainder = lez * ((integer - tensor) % integer)
    remainder = pos_remainder + neg_remainder
    quotient = tensor / integer
    correction = (2 * remainder > integer).long()
    return quotient + tensor.sign() * correction


class FixedPointEncoder:
    """Encoder that encodes long or float tensors into scaled base tensors."""

    def __init__(self, precision_bits=16):
        self._scale = int(2 ** precision_bits)

    def encode(self, x):
        """Helper function to wrap data if needed"""
        if isinstance(x, EncryptedTensor):
            return x
        elif isinstance(x, int) or isinstance(x, float):
            # Squeeze in order to get a 0-dim tensor with value `x`
            return torch.LongTensor([self._scale * x]).squeeze()
        elif is_float_tensor(x):
            return (self._scale * x).long()
        elif isinstance(x, np.ndarray):
            return self._scale * torch.from_numpy(x).long()
        # For integer types cast to long prior to scaling to avoid overflow.
        elif is_int_tensor(x):
            return self._scale * x.long()
        elif isinstance(x, np.ndarray):
            return self._scale * torch.from_numpy(x).long()
        elif torch.is_tensor(x):
            raise TypeError("Cannot encode input with dtype %s" % x.dtype)
        else:
            raise TypeError("Unknown tensor type: %s." % type(x))

    def decode(self, tensor, to_float=True):
        """Helper function that decodes from scaled tensor"""
        assert is_int_tensor(tensor), "input must be a LongTensor"
        if to_float:
            correction = (tensor < 0).long()
            dividend = tensor / self._scale - correction
            remainder = tensor % self._scale
            remainder += (remainder == 0).long() * self._scale * correction

            tensor = dividend.float() + remainder.float() / self._scale
        else:
            tensor = nearest_integer_division(tensor, self._scale)

        return tensor

    @property
    def scale(self):
        return self._scale
