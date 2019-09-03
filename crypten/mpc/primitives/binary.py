#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# dependencies:
import torch
from crypten import comm
from crypten.common.rng import generate_kbit_random_tensor
from crypten.common.tensor_types import is_int_tensor
from crypten.cryptensor import CrypTensor
from crypten.encoder import FixedPointEncoder

from . import beaver, circuit


SENTINEL = -1


# MPC tensor where shares are XOR-sharings.
class BinarySharedTensor(CrypTensor):
    """
        Encrypted tensor object that uses binary sharing to perform computations.

        Binary shares are computed by splitting each value of the input tensor
        into n separate random values that xor together to the input tensor value,
        where n is the number of parties present in the protocol (world_size).
    """

    def __init__(self, tensor=None, size=None, src=0):
        if src == SENTINEL:
            return

        assert is_int_tensor(tensor), "input must be an integer tensor"

        #  Assume 0 bits of precision unless encoder is set outside of init
        self.encoder = FixedPointEncoder(precision_bits=0)
        if tensor is not None:
            tensor = self.encoder.encode(tensor)
            size = tensor.size()

        # Generate Psuedo-random Sharing of Zero and add source's tensor
        self._tensor = BinarySharedTensor.PRZS(size)._tensor
        if self.rank == src:
            assert tensor is not None, "Source must provide a data tensor"
            self._tensor ^= tensor

    @staticmethod
    def from_shares(share, src=0):
        """Generate an AdditiveSharedTensor from a share from each party"""
        result = BinarySharedTensor(src=SENTINEL)
        result._tensor = share
        result.encoder = FixedPointEncoder(precision_bits=0)
        return result

    @staticmethod
    def PRZS(*size):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. Therefore, each party holds
        two numbes. A zero sharing is found by having each party xor their two
        numbers together.
        """
        tensor = BinarySharedTensor(src=SENTINEL)
        current_share = generate_kbit_random_tensor(*size, generator=comm.g0)
        next_share = generate_kbit_random_tensor(*size, generator=comm.g1)
        tensor._tensor = current_share ^ next_share
        return tensor

    @property
    def rank(self):
        return comm.get_rank()

    def shallow_copy(self):
        """Create a shallow copy"""
        result = BinarySharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor
        return result

    def __repr__(self):
        return "%s BinarySharedTensor" % str(tuple(self.size()))

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def XOR_(self, y):
        """Bitwise XOR operator (element-wise) in place"""
        if torch.is_tensor(y) or isinstance(y, int):
            if self.rank == 0:
                self._tensor ^= y
        elif isinstance(y, BinarySharedTensor):
            self._tensor ^= y._tensor
        else:
            raise TypeError("Cannot XOR %s with %s." % (type(y), type(self)))
        return self

    def XOR(self, y):
        """Bitwise XOR operator (element-wise)"""
        return self.clone().XOR_(y)

    def AND_(self, y):
        """Bitwise AND operator (element-wise) in place"""
        if torch.is_tensor(y) or isinstance(y, int):
            self._tensor &= y
        elif isinstance(y, BinarySharedTensor):
            self._tensor.data = beaver.AND(self, y)._tensor.data
        else:
            raise TypeError("Cannot AND %s with %s." % (type(y), type(self)))
        return self

    def AND(self, y):
        """Bitwise AND operator (element-wise)"""
        return self.clone().AND_(y)

    def OR_(self, y):
        """Bitwise OR operator (element-wise) in place"""
        xor_result = self ^ y
        return self.AND_(y).XOR_(xor_result)

    def OR(self, y):
        """Bitwise OR operator (element-wise)"""
        return self.AND(y) ^ self ^ y

    def __invert__(self):
        """Bitwise NOT operator (element-wise)"""
        result = self.clone()
        if result.rank == 0:
            result._tensor ^= -1
        return result

    def lshift_(self, value):
        """Left shift elements by `value` bits"""
        assert isinstance(value, int), "lshift must take an integer argument."
        self._tensor <<= value
        return self

    def lshift(self, value):
        """Left shift elements by `value` bits"""
        return self.clone().lshift_(value)

    def rshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "rshift must take an integer argument."
        self._tensor >>= value
        return self

    def rshift(self, value):
        """Right shift elements by `value` bits"""
        return self.clone().rshift_(value)

    # Circuits
    def add(self, y):
        """Compute [self] + [y] for xor-sharing"""
        return circuit.add(self, y)

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if torch.is_tensor(value) or isinstance(value, list):
            value = BinarySharedTensor(value)
        assert isinstance(
            value, BinarySharedTensor
        ), "Unsupported input type %s for __setitem__" % type(value)
        self._tensor.__setitem__(index, value._tensor)

    @staticmethod
    def stack(seq, *args, **kwargs):
        """Stacks a list of tensors along a given dimension"""
        assert isinstance(seq, list), "Stack input must be a list"
        assert isinstance(
            seq[0], BinarySharedTensor
        ), "Sequence must contain BinarySharedTensors"
        result = seq[0].shallow_copy()
        result._tensor = torch.stack(
            [BinarySharedTensor._tensor for BinarySharedTensor in seq], *args, **kwargs
        )
        return result

    def sum(self, dim=None):
        """Add all tensors along a given dimension using a log-reduction"""
        if dim is None:
            x = self.flatten()
        else:
            x = self.transpose(0, dim)

        # Add all BinarySharedTensors
        while x.size(0) > 1:
            extra = None
            if x.size(0) % 2 == 1:
                extra = x[0]
                x = x[1:]
            x0 = x[: (x.size(0) // 2)]
            x1 = x[(x.size(0) // 2) :]
            x = x0 + x1
            if extra is not None:
                x._tensor = torch.cat([x._tensor, extra._tensor.unsqueeze(0)])

        if dim is None:
            x = x.squeeze()
        else:
            x = x.transpose(0, dim).squeeze(dim)
        return x

    def cumsum(self, *args, **kwargs):
        raise NotImplementedError("BinarySharedTensor cumsum not implemented")

    def trace(self, *args, **kwargs):
        raise NotImplementedError("BinarySharedTensor trace not implemented")

    def reveal(self):
        """Get plaintext without any downscaling"""
        shares = comm.all_gather(self._tensor)
        result = shares[0]
        for x in shares[1:]:
            result = result ^ x
        return result

    def get_plain_text(self):
        """Decrypt the tensor"""
        return self.encoder.decode(self.reveal())

    # Bitwise operators
    __xor__ = XOR
    __and__ = AND
    __or__ = OR
    __lshift__ = lshift
    __rshift__ = rshift

    # In-place bitwise operators
    __ixor__ = XOR_
    __iand__ = AND_
    __ior__ = OR_
    __ilshift__ = lshift_
    __irshift__ = rshift_

    # Reversed boolean operations
    __rxor__ = __xor__
    __rand__ = __and__
    __ror__ = __or__


REGULAR_FUNCTIONS = [
    "clone",
    "__getitem__",
    "index_select",
    "view",
    "flatten",
    "t",
    "transpose",
    "unsqueeze",
    "squeeze",
    "repeat",
    "narrow",
    "expand",
    "roll",
    "unfold",
    "flip",
    "reshape",
    "gather",
    "take",
    "index_select",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, function_name)(*args, **kwargs)
        return result

    setattr(BinarySharedTensor, function_name, regular_func)


def _add_property_function(function_name):
    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    setattr(BinarySharedTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
