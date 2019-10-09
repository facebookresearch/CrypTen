#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm

# dependencies:
import torch
from crypten.common.rng import generate_kbit_random_tensor
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
        assert (
            isinstance(src, int) and src >= 0 and src < comm.get().get_world_size()
        ), "invalid tensor source"

        #  Assume 0 bits of precision unless encoder is set outside of init
        self.encoder = FixedPointEncoder(precision_bits=0)
        if tensor is not None:
            tensor = self.encoder.encode(tensor)
            size = tensor.size()

        # Generate Psuedo-random Sharing of Zero and add source's tensor
        self.share = BinarySharedTensor.PRZS(size).share
        if self.rank == src:
            assert tensor is not None, "Source must provide a data tensor"
            if hasattr(tensor, "src"):
                assert (
                    tensor.src == src
                ), "Source of data tensor must match source of encryption"
            self.share ^= tensor

    @staticmethod
    def from_shares(share, src=0):
        """Generate a BinarySharedTensor from a share from each party"""
        result = BinarySharedTensor(src=SENTINEL)
        result.share = share
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
        current_share = generate_kbit_random_tensor(*size, generator=comm.get().g0)
        next_share = generate_kbit_random_tensor(*size, generator=comm.get().g1)
        tensor.share = current_share ^ next_share
        return tensor

    @property
    def rank(self):
        return comm.get().get_rank()

    @property
    def share(self):
        """Returns underlying _tensor"""
        return self._tensor

    @share.setter
    def share(self, value):
        """Sets _tensor to value"""
        self._tensor = value

    def shallow_copy(self):
        """Create a shallow copy"""
        result = BinarySharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result.share = self.share
        return result

    def __repr__(self):
        return f"BinarySharedTensor({self.share})"

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __ixor__(self, y):
        """Bitwise XOR operator (element-wise) in place"""
        if torch.is_tensor(y) or isinstance(y, int):
            if self.rank == 0:
                self.share ^= y
        elif isinstance(y, BinarySharedTensor):
            self.share ^= y.share
        else:
            raise TypeError("Cannot XOR %s with %s." % (type(y), type(self)))
        return self

    def __xor__(self, y):
        """Bitwise XOR operator (element-wise)"""
        result = self.clone()
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif torch.is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__ixor__(y)

    def __iand__(self, y):
        """Bitwise AND operator (element-wise) in place"""
        if torch.is_tensor(y) or isinstance(y, int):
            self.share &= y
        elif isinstance(y, BinarySharedTensor):
            self.share.data = beaver.AND(self, y).share.data
        else:
            raise TypeError("Cannot AND %s with %s." % (type(y), type(self)))
        return self

    def __and__(self, y):
        """Bitwise AND operator (element-wise)"""
        result = self.clone()
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif torch.is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__iand__(y)

    def __ior__(self, y):
        """Bitwise OR operator (element-wise) in place"""
        xor_result = self ^ y
        return self.__iand__(y).__ixor__(xor_result)

    def __or__(self, y):
        """Bitwise OR operator (element-wise)"""
        return self.__and__(y) ^ self ^ y

    def __invert__(self):
        """Bitwise NOT operator (element-wise)"""
        result = self.clone()
        if result.rank == 0:
            result.share ^= -1
        return result

    def lshift_(self, value):
        """Left shift elements by `value` bits"""
        assert isinstance(value, int), "lshift must take an integer argument."
        self.share <<= value
        return self

    def lshift(self, value):
        """Left shift elements by `value` bits"""
        return self.clone().lshift_(value)

    def rshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "rshift must take an integer argument."
        self.share >>= value
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
        self.share.__setitem__(index, value.share)

    @staticmethod
    def stack(seq, *args, **kwargs):
        """Stacks a list of tensors along a given dimension"""
        assert isinstance(seq, list), "Stack input must be a list"
        assert isinstance(
            seq[0], BinarySharedTensor
        ), "Sequence must contain BinarySharedTensors"
        result = seq[0].shallow_copy()
        result.share = torch.stack(
            [BinarySharedTensor.share for BinarySharedTensor in seq], *args, **kwargs
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
                x.share = torch.cat([x.share, extra.share.unsqueeze(0)])

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
        shares = comm.get().all_gather(self.share)
        result = shares[0]
        for x in shares[1:]:
            result = result ^ x
        return result

    def get_plain_text(self):
        """Decrypt the tensor"""
        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.nelement() < 1:
            return torch.empty(self.share.size())
        return self.encoder.decode(self.reveal())

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or BinarySharedTensor): when True yield self,
                otherwise yield y. Note condition is not bitwise.
            y (torch.tensor or BinarySharedTensor): selected when condition is
                False.

        Returns: BinarySharedTensor or torch.tensor.
        """
        if torch.is_tensor(condition):
            condition = condition.long()
            is_binary = ((condition == 1) | (condition == 0)).all()
            assert is_binary, "condition values must be 0 or 1"
            # -1 mult expands 0 into binary 00...00 and 1 into 11...11
            condition_expanded = -condition
            y_masked = y & (~condition_expanded)
        elif isinstance(condition, BinarySharedTensor):
            condition_expanded = condition.clone()
            # -1 mult expands binary while & 1 isolates first bit
            condition_expanded.share = -(condition_expanded.share & 1)
            # encrypted tensor must be first operand
            y_masked = (~condition_expanded) & y
        else:
            msg = f"condition {condition} must be torch.bool, or BinarySharedTensor"
            raise ValueError(msg)

        return (self & condition_expanded) ^ y_masked

    def scatter_(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        if torch.is_tensor(src):
            src = BinarySharedTensor(src)
        assert isinstance(
            src, BinarySharedTensor
        ), "Unrecognized scatter src type: %s" % type(src)
        self.share.scatter_(dim, index, src.share)
        return self

    def scatter(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        result = self.clone()
        return result.scatter_(self, dim, index, src)

    # Bitwise operators
    __lshift__ = lshift
    __rshift__ = rshift

    # In-place bitwise operators
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
    "scatter",
    "take",
    "split",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result.share = getattr(result.share, function_name)(*args, **kwargs)
        return result

    setattr(BinarySharedTensor, function_name, regular_func)


def _add_property_function(function_name):
    def property_func(self, *args, **kwargs):
        return getattr(self.share, function_name)(*args, **kwargs)

    setattr(BinarySharedTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
