#!/usr/bin/env python3

from enum import IntEnum

from .primitives.arithmetic import ArithmeticSharedTensor
from .primitives.binary import BinarySharedTensor


class ptype(IntEnum):
    """Enumeration defining the private type attributes of encrypted tensors"""

    arithmetic = 0
    binary = 1

    def to_tensor(self):
        if self.value == 0:
            return ArithmeticSharedTensor
        elif self.value == 1:
            return BinarySharedTensor
        else:
            raise ValueError("Cannot convert %s to encrypted tensor" % (self.name))
