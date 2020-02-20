#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from crypten.cryptensor import CrypTensor
from crypten.common.tensor_types import is_float_tensor, is_int_tensor


class FixedPointEncoder:
    """Encoder that encodes long or float tensors into scaled integer tensors."""

    __default_precision_bits = 16

    def __init__(self, precision_bits=None):
        if precision_bits is None:
            precision_bits = FixedPointEncoder.__default_precision_bits
        self._scale = int(2 ** precision_bits)

    def encode(self, x):
        """Helper function to wrap data if needed"""
        if isinstance(x, CrypTensor):
            return x
        elif isinstance(x, int) or isinstance(x, float):
            # Squeeze in order to get a 0-dim tensor with value `x`
            return torch.LongTensor([self._scale * x]).squeeze()
        elif isinstance(x, list):
            return torch.FloatTensor(x).mul_(self._scale).long()
        elif is_float_tensor(x):
            return (self._scale * x).long()
        # For integer types cast to long prior to scaling to avoid overflow.
        elif is_int_tensor(x):
            return self._scale * x.long()
        elif isinstance(x, np.ndarray):
            return self._scale * torch.from_numpy(x).long()
        elif torch.is_tensor(x):
            raise TypeError("Cannot encode input with dtype %s" % x.dtype)
        else:
            raise TypeError("Unknown tensor type: %s." % type(x))

    def decode(self, tensor, dtype=torch.float32):
        """Helper function that decodes from scaled tensor"""
        assert is_int_tensor(tensor), "input must be a LongTensor"

        result = tensor.to(dtype)
        if self._scale > 1:
            result /= self._scale

        return result

    @property
    def scale(self):
        return self._scale

    @classmethod
    def set_default_precision(cls, precision_bits):
        assert (
            isinstance(precision_bits, int)
            and precision_bits >= 0
            and precision_bits < 64
        ), "precision must be a positive integer less than 64"
        cls.__default_precision_bits = precision_bits
