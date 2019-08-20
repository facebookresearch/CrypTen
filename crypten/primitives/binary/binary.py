#!/usr/bin/env python3

import crypten.common.bitwise as bitwise
import crypten.common.constants as constants

# dependencies:
import torch
from crypten import comm
from crypten.common import EncryptedTensor, FixedPointEncoder
from crypten.common.sharing import xor_share
from crypten.common.tensor_types import is_int_tensor

from .beaver import Beaver
from .circuit import Circuit


SENTINEL = -1


# MPC tensor where shares are XOR-sharings.
class BinarySharedTensor(EncryptedTensor):
    """
        Encrypted tensor object that uses binary sharing to perform computations.

        Binary shares are computed by splitting each value of the input tensor
        into n separate random values that xor together to the input tensor value,
        where n is the number of parties present in the protocol (world_size).
    """

    def __init__(self, tensor=None, size=None, src=0):
        if src == SENTINEL:
            return

        # _rank indicates the rank of the current processes
        # _src indicates the rank of the source process that will provide data shares
        self._rank = comm.get_rank()
        self._src = src

        #  Assume 0 bits of precision unless encoder is set outside of init
        self.encoder = FixedPointEncoder(precision_bits=0)

        if self._rank == self._src:
            assert tensor is not None, "Data source must supply an input tensor."
            if isinstance(tensor, (list, int)):
                tensor = torch.LongTensor(tensor)

            assert is_int_tensor(tensor), "input must be an integer tensor"
            shares = xor_share(tensor, num_parties=comm.get_world_size())
            self._tensor = comm.scatter(shares, src)
        else:
            # TODO: Remove this adapt tests to use size arg
            if isinstance(tensor, int):
                tensor = torch.LongTensor(tensor)
            size = tensor.size()
            self._tensor = comm.scatter(None, src, size=size)

    def shallow_copy(self):
        """Create a shallow copy"""
        result = BinarySharedTensor(src=SENTINEL)
        result._rank = self._rank
        result._src = self._src
        result.encoder = self.encoder
        result._tensor = self._tensor
        return result

    def XOR_(self, y):
        """Bitwise XOR operator (element-wise)"""
        if torch.is_tensor(y) or isinstance(y, int):
            if self._rank == 0:
                self._tensor ^= y
        elif isinstance(y, BinarySharedTensor):
            self._tensor ^= y._tensor
        else:
            raise TypeError("Cannot XOR %s with %s." % (type(y), type(self)))
        return self

    def XOR(self, y):
        """Bitwise XOR operator (element-wise)"""
        return self.clone().XOR_(y)

    def AND_(self, y, bits=constants.K):
        if torch.is_tensor(y) or isinstance(y, int):
            self._tensor &= y
        elif isinstance(y, BinarySharedTensor):
            self = Beaver.AND(self, y, bits=bits)
        else:
            raise TypeError("Cannot AND %s with %s." % (type(y), type(self)))
        return self

    def AND(self, y, bits=constants.K):
        """Bitwise AND operator (element-wise)"""
        return self.clone().AND_(y, bits=bits)

    def OR_(self, y, bits=constants.K):
        self = self.AND(y, bits=bits) ^ self ^ y
        return self

    def OR(self, y, bits=constants.K):
        return self.AND(y, bits=bits) ^ self ^ y

    def neg_(self):
        self = Circuit.add(~self, 1)
        return self

    def neg(self):
        """Returns -self"""
        return Circuit.add(~self, 1)

    def invert_(self):
        """Bitwise NOT operator (element-wise)"""
        if self._rank == 0:
            self._tensor ^= -1
        return self

    def invert(self):
        """Bitwise NOT operator (element-wise)"""
        return self.clone().invert_()

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

    def get_bit(self, index):
        """Get an individual bit of each value"""
        result = self.shallow_copy()
        result._tensor = (result._tensor >> index) & 1
        return result

    def set_bit(self, index, value):
        """Set an individual bit of each value"""
        if torch.is_tensor(value):
            if self.rank == 0:
                self._tensor = bitwise.set_bit(self._tensor, index, value)
            else:
                self._tensor = bitwise.set_bit(self._tensor, index, 0)
        elif isinstance(value, BinarySharedTensor):
            self._tensor = bitwise.set_bit(self._tensor, index, value._tensor)
        else:
            raise TypeError("Cannot set_bit with type: %s" % type(value))
        return self

    # Circuits
    def add(self, y):
        """Compute [self] + [y] for xor-sharing"""
        return Circuit.add(self, y)

    def lt(self, y):
        """Compute [self] < [y] for xor-sharing"""
        result = Circuit.lt(self, y)
        return result

    def eq(self, y):
        """Compute [self] == [y] for xor-sharing"""
        result = Circuit.eq(self, y)
        return result

    def gt(self, y):
        """Compute [self] > [y] for xor-sharing"""
        result = Circuit.lt(y, self)
        return result

    def le(self, y):
        """Compute [self] <= [y] for xor-sharing"""
        return self.gt(y) ^ 1

    def ge(self, y):
        """Compute [self] >= [y] for xor-sharing"""
        return self.lt(y) ^ 1

    def ne(self, y):
        """Compute [self] != [y] for xor-sharing"""
        return self.eq(y) ^ 1

    # TODO: Correct the implementations for min, max, argmin, argmax
    '''
    def argmax(self):
        """Returns 1 for the element that has the highest value"""
        assert self.dim() == 1, 'Argmax only implemented for 1D tensors'

        def _toeplitz(vector):
            size = len(vector)
            matrix = vector.repeat(size, 1)
            for i in range(1, size):
                matrix[i] = matrix[i].roll(i)

            return matrix

        a = self.clone()
        b = a.clone()

        a._tensor = a._tensor.repeat(len(a._tensor) - 1, 1)
        b._tensor = _toeplitz(b._tensor)[1:]

        result = a.ge(b)

        # Compute an AND along each column
        def _fold_and(x):
            rows = x.size(0)
            half_rows = (rows + 1) // 2
            if rows % 2 == 1:
                y = x._tensor[0].unsqueeze(0)
                x0 = x[1:half_rows]
                x1 = x[half_rows:]
            else:
                y = None
                x0 = x[:half_rows]
                x1 = x[half_rows:]

            result = x0.AND(x1, bits=1)
            if y is not None:
                result._tensor = BaseTensor.cat([y, result._tensor])
            return result

        while result.size(0) > 1:
            result = _fold_and(result)

        result._tensor = result._tensor.squeeze()
        return result

    def argmin(self):
        """Returns 1 for the element that has the lowest value"""
        return (-self).argmax()

    def max(self):
        """Compute the max of a tensor's elements (or along a given dimension)"""
        amax = self.argmax() >> constants.PRECISION
        amax._tensor *= -1  # turns 00...01 into 11...11
        result = self.AND(amax).xor_sum()
        return result

    def min(self, **kwargs):
        """Compute the min of a tensor's elements (or along a given dimension)"""
        amax = self.argmin() >> constants.PRECISION
        amax._tensor *= -1  # turns 00...01 into 11...11
        result = self.AND(amax).xor_sum()
        return result

    def xor_sum(self):
        result = BaseTensor([0])
        x = self.clone()
        x._tensor = x._tensor.flatten()
        for elem in x._tensor:
            result ^= elem
        x._tensor = result
        return x
    '''

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
            x = self.view(-1)
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
            x = x.squeeze(0)
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
        return self.reveal()

    __xor__ = XOR
    __ixor__ = XOR_
    __or__ = OR
    __and__ = AND
    __iand__ = AND_
    __invert__ = invert
    __lshift__ = lshift
    __rshift__ = rshift
    __ge__ = ge
    __gt__ = gt
    __le__ = le
    __lt__ = lt
    __eq__ = eq
    __ne__ = ne

    # Add reversed boolean operations
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    @staticmethod
    def print_communication_stats():
        comm.print_communication_stats()

    @staticmethod
    def reset_communication_stats():
        comm.reset_communication_stats()
