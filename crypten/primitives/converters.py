#!/usr/bin/env python3

from crypten import comm
from crypten.common import constants

from .arithmetic import ArithmeticSharedTensor
from .binary import Beaver, BinarySharedTensor


def _A2B(arithmetic_tensor):
    binary_tensor = BinarySharedTensor.stack(
        [
            BinarySharedTensor(arithmetic_tensor._tensor, src=i)
            for i in range(comm.get_world_size())
        ]
    )
    binary_tensor = binary_tensor.sum(dim=0)
    binary_tensor.encoder = arithmetic_tensor.encoder
    return binary_tensor


def _B2A(binary_tensor, bits=constants.K):
    arithmetic_tensor = 0
    for i in range(bits):
        # TODO: Move Beaver.B2A_single_bit to a more appropriate location
        arithmetic_bit = Beaver.B2A_single_bit(binary_tensor.get_bit(i))
        arithmetic_tensor += arithmetic_bit * (2 ** i)
    arithmetic_tensor.encoder = binary_tensor.encoder
    arithmetic_tensor *= arithmetic_tensor.encoder._scale
    return arithmetic_tensor


def convert(tensor, ptype):
    if isinstance(tensor, ptype):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == BinarySharedTensor:
        return _A2B(tensor)
    elif isinstance(tensor, BinarySharedTensor) and ptype == ArithmeticSharedTensor:
        # Assuming GMW results are always 1-bit
        return _B2A(tensor, bits=1)
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))
