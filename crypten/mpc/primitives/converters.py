#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder

from ..ptype import ptype as Ptype
from . import beaver
from .arithmetic import ArithmeticSharedTensor
from .binary import BinarySharedTensor


def _A2B(arithmetic_tensor):
    binary_tensor = BinarySharedTensor.stack(
        [
            BinarySharedTensor(arithmetic_tensor.share, src=i)
            for i in range(comm.get().get_world_size())
        ]
    )
    binary_tensor = binary_tensor.sum(dim=0)
    binary_tensor.encoder = arithmetic_tensor.encoder
    return binary_tensor


def _B2A(binary_tensor, precision=None, bits=None):
    if bits is None:
        bits = torch.iinfo(torch.long).bits

    arithmetic_tensor = 0
    for i in range(bits):
        binary_bit = binary_tensor & 1
        arithmetic_bit = beaver.B2A_single_bit(binary_bit)
        # avoids long integer overflow since 2 ** 63 is out of range
        # (aliases to -2 ** 63)
        if i == 63:
            arithmetic_tensor += arithmetic_bit * (-2 ** 63)
        else:
            arithmetic_tensor += arithmetic_bit * (2 ** i)
        binary_tensor >>= 1
    arithmetic_tensor.encoder = FixedPointEncoder(precision_bits=precision)
    scale = arithmetic_tensor.encoder._scale // binary_tensor.encoder._scale
    arithmetic_tensor *= scale
    return arithmetic_tensor


def convert(tensor, ptype, **kwargs):
    tensor_name = ptype.to_tensor()
    if isinstance(tensor, tensor_name):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == Ptype.binary:
        return _A2B(tensor)
    elif isinstance(tensor, BinarySharedTensor) and ptype == Ptype.arithmetic:
        return _B2A(tensor, **kwargs)
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))
