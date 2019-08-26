#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class EncryptedTensor:
    """
        Encrypted tensor type that is private and cannot be shown to the outside world.
    """

    def abs(self):
        raise NotImplementedError("abs is not implemented")

    def __abs__(self):
        return self.abs()

    def pow(self):
        raise NotImplementedError("pow is not implemented")

    def __pow__(self, tensor):
        return self.pow(tensor)

    def __rpow__(self, scalar):
        raise NotImplementedError("__rpow__ is not implemented")

    def __init__(self):
        raise NotImplementedError("Cannot instantiate an EncryptedTensor")

    def get_plain_text(self):
        """Decrypts the encrypted tensor."""
        raise NotImplementedError("get_plain_text is not implemented")

    def shallow_copy(self):
        """Creates a shallow_copy of a tensor"""
        raise NotImplementedError("shallow_copy is not implemented")

    def add_(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        raise NotImplementedError("add_ is not implemented")

    def add(self, tensor):
        """Adds tensor to this tensor."""
        raise NotImplementedError("add is not implemented")

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def sub_(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        raise NotImplementedError("sub_ is not implemented")

    def sub(self, tensor):
        """Subtracts tensor from this tensor."""
        raise NotImplementedError("sub is not implemented")

    def __sub__(self, tensor):
        """Subtracts tensor from this tensor."""
        return self.sub(tensor)

    def __rsub__(self, tensor):
        """Subtracts self from tensor."""
        return -self + tensor

    def __isub__(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        return self.sub_(tensor)

    def mul_(self, tensor):
        """Element-wise multiply with a tensor (in-place)."""
        raise NotImplementedError("mul_ is not implemented")

    def mul(self, tensor):
        """Element-wise multiply with a tensor."""
        raise NotImplementedError("mul is not implemented")

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def div_(self, scalar):
        """Element-wise divide by a tensor (in-place)."""
        raise NotImplementedError("div_ is not implemented")

    def div(self, scalar):
        """Element-wise divide by a tensor."""
        raise NotImplementedError("div is not implemented")

    def __div__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def neg(self):
        """Negative value of a tensor"""
        raise NotImplementedError("neg is not implemented")

    def neg_(self):
        """Negative value of a tensor (in-place)"""
        raise NotImplementedError("neg_ is not implemented")

    def __neg__(self):
        return self.neg()

    def matmul(self, tensor):
        """Perform matrix multiplication using some tensor"""
        raise NotImplementedError("matmul is not implemented")

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def eq(self, tensor):
        """Element-wise equality"""
        raise NotImplementedError("eq is not implemented")

    def __eq__(self, tensor):
        """Element-wise equality"""
        return self.eq(tensor)

    def ne(self, tensor):
        """Element-wise inequality"""
        raise NotImplementedError("ne is not implemented")

    def __ne__(self, tensor):
        """Element-wise inequality"""
        return self.ne(tensor)

    def ge(self, tensor):
        """Element-wise greater than or equal to"""
        raise NotImplementedError("ge is not implemented")

    def __ge__(self, tensor):
        """Element-wise greater than or equal to"""
        return self.ge(tensor)

    def gt(self, tensor):
        """Element-wise greater than"""
        raise NotImplementedError("gt is not implemented")

    def __gt__(self, tensor):
        """Element-wise greater than"""
        return self.gt(tensor)

    def le(self, tensor):
        """Element-wise less than or equal to"""
        raise NotImplementedError("le is not implemented")

    def __le__(self, tensor):
        """Element-wise less than or equal to"""
        return self.le(tensor)

    def lt(self, tensor):
        """Element-wise less than"""
        raise NotImplementedError("lt is not implemented")

    def __lt__(self, tensor):
        """Element-wise less than"""
        return self.lt(tensor)

    def dot(self, tensor, weights=None):
        """Perform (weighted) inner product with plain or cipher text."""
        raise NotImplementedError("dot is not implemented")

    def onnx_gather(self, index, dimension):
        """Gather entries of tensor along a dimension according to indices"""
        raise NotImplementedError("onnx_gather is not implemented")


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
    "expand",
    "roll",
    "unfold",
    "take",
    "flip",
    "trace",
    "sum",
    "cumsum",
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

    setattr(EncryptedTensor, function_name, regular_func)


def _add_property_function(function_name):
    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    setattr(EncryptedTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
