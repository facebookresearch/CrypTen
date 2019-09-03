#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch
from crypten.common.util import pool_reshape

from ..cryptensor import CrypTensor
from .primitives.converters import convert
from .ptype import ptype as Ptype


def mode(ptype, inplace=False):
    if inplace:

        def function_wrapper(func):
            def convert_wrapper(self, *args, **kwargs):
                self._tensor = convert(self._tensor, ptype)
                self.ptype = ptype
                self = func(self, *args, **kwargs)
                return self

            return convert_wrapper

    else:

        def function_wrapper(func):
            def convert_wrapper(self, *args, **kwargs):
                result = self.to(ptype)
                return func(result, *args, **kwargs)

            return convert_wrapper

    return function_wrapper


class MPCTensor(CrypTensor):
    def __init__(self, input, ptype=Ptype.arithmetic, *args, **kwargs):
        if input is None:
            return
        tensor_name = ptype.to_tensor()
        self._tensor = tensor_name(input, *args, **kwargs)
        self.ptype = ptype

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    def shallow_copy(self):
        """Create a shallow copy of the input tensor"""
        result = MPCTensor(None)
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    # Handle share types and conversions
    def to(self, ptype, **kwargs):
        """Converts self._tensor to the given ptype"""
        retval = self.clone()
        if retval.ptype == ptype:
            return retval
        retval._tensor = convert(self._tensor, ptype, **kwargs)
        retval.ptype = ptype
        return retval

    def arithmetic(self):
        """Converts self._tensor to arithmetic secret sharing"""
        return self.to(Ptype.arithmetic)

    def binary(self):
        """Converts self._tensor to binary secret sharing"""
        return self.to(Ptype.binary)

    def get_plain_text(self):
        """Decrypt the tensor"""
        return self._tensor.get_plain_text()

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if not isinstance(value, MPCTensor):
            value = MPCTensor(value, ptype=self.ptype)
        self._tensor.__setitem__(index, value._tensor)

    def bernoulli(self):
        """Draws a random tensor of {0, 1} with given probabilities"""
        return self > crypten.mpc.rand(self.size())

    # Comparators
    @mode(Ptype.binary)
    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        result = (self >> shift).to(Ptype.arithmetic, bits=1)
        return result * result._tensor.encoder._scale

    @mode(Ptype.arithmetic)
    def ge(self, y):
        """Returns self >= y"""
        return 1 - self.lt(y)

    @mode(Ptype.arithmetic)
    def gt(self, y):
        """Returns self > y"""
        return (-self + y)._ltz()

    @mode(Ptype.arithmetic)
    def le(self, y):
        """Returns self <= y"""
        return 1 - self.gt(y)

    @mode(Ptype.arithmetic)
    def lt(self, y):
        """Returns self < y"""
        return (self - y)._ltz()

    @mode(Ptype.arithmetic)
    def eq(self, y):
        """Returns self == y"""
        return self.ge(y) - self.gt(y)

    @mode(Ptype.arithmetic)
    def ne(self, y):
        """Returns self != y"""
        return 1 - self.eq(y)

    @mode(Ptype.arithmetic)
    def sign(self):
        """Computes the sign value of a tensor (0 is considered positive)"""
        return 2 * (self >= 0) - 1

    @mode(Ptype.arithmetic)
    def abs(self):
        """Computes the absolute value of a tensor"""
        return self * self.sign()

    @mode(Ptype.arithmetic)
    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        return self * (self > 0)

    # max / min-related functions
    def _argmax_helper(self):
        """Returns 1 for all elements that have the highest value in each row"""
        row_length = self.size(-1) if self.size(-1) > 1 else 2

        # Copy each row (length - 1) times to compare to each other row
        a = self.expand(row_length - 1, *self.size())

        # Generate cyclic permutations for each row
        b = crypten.mpc.stack(
            [self.roll(i + 1, dims=-1) for i in range(row_length - 1)]
        )

        # Sum of columns with all 1s will have value equal to (length - 1).
        # Using >= since it requires 1-fewer comparrison than !=
        result = (a >= b).sum(dim=0)
        return result >= (row_length - 1)

    @mode(Ptype.arithmetic)
    def argmax(self, dim=None, one_hot_required=True):
        """Returns a one-hot vector with a 1 entry at a maximum value.

        If multiple values are equal to the maximum, it will choose one randomly,
        then ties will be broken (randomly) if one_hot_required is True.
        Otherwise, all indices with maximal inputs will be return a 1.
        """
        if self.dim() == 0:
            return MPCTensor(torch.zeros(())) + 1
        if dim is None:
            input = self.flatten()
        else:
            input = self.transpose(dim, -1)

        result = input._argmax_helper()

        # Multiply by a random permutation to give each maximum a random priority
        if one_hot_required:
            result *= crypten.mpc.randperm(input.size())
            result = result._argmax_helper()

        if dim is None:
            return result.view(self.size())
        else:
            return result.transpose(dim, -1)

    @mode(Ptype.arithmetic)
    def argmin(self, **kwargs):
        """Returns a one-hot vector with a 1 entry at a minimum value. If multiple
        values are equal to the minimum, it will choose one randomly"""
        return (-self).argmax(**kwargs)

    @mode(Ptype.arithmetic)
    def max(self, dim=None, **kwargs):
        """Compute the max of a tensor's elements (or along a given dimension)"""
        if dim is None:
            return self.mul(self.argmax(**kwargs)).sum()
        else:
            result = self * self.argmax(dim=dim, **kwargs)
            return result.sum(dim=dim)

    @mode(Ptype.arithmetic)
    def min(self, **kwargs):
        """Compute the min of a tensor's elements (or along a given dimension)"""
        return -((-self).max(**kwargs))

    @mode(Ptype.arithmetic)
    def max_pool2d(self, kernel_size, padding=None, stride=None):
        """Perform a max pooling on each 2D matrix of the given tensor"""
        max_input = self.shallow_copy()
        max_input._tensor._tensor, output_size = pool_reshape(
            self._tensor._tensor,
            kernel_size,
            padding=padding,
            stride=stride,
            # padding with extremely negative values to avoid choosing pads
            # -2 ** 40 is acceptable since it is lower than the supported range
            # which is -2 ** 32 because multiplication can otherwise fail.
            pad_value=(-2 ** 40),
        )
        max_vals = max_input.max(dim=-1)
        result = max_vals.view(output_size)
        return result

    # Logistic Functions
    @mode(Ptype.arithmetic)
    def sigmoid(self, reciprocal_method="log"):
        """Computes the sigmoid function on the input value
                sigmoid(x) = (1 + exp(-x))^{-1}

        For numerical stability, we compute this by:
                sigmoid(x) = (sigmoid(|x|) - 0.5) * sign(x) + 0.5
        """
        sign = self.sign()
        x = self * sign
        result = (1 + (-x).exp()).reciprocal(method=reciprocal_method, log_iters=2)
        return (result - 0.5) * sign + 0.5

    @mode(Ptype.arithmetic)
    def tanh(self, reciprocal_method="log"):
        """Computes tanh from the sigmoid function:
            tanh(x) = 2 * sigmoid(2 * x) - 1
        """
        return (self * 2).sigmoid(reciprocal_method=reciprocal_method) * 2 - 1

    @mode(Ptype.arithmetic)
    def pad(self, pad, mode="constant", value=0):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            result._tensor = self._tensor.pad(pad, mode=mode, value=value._tensor)
        else:
            result._tensor = self._tensor.pad(pad, mode=mode, value=value)
        return result

    # Approximations:
    def div(self, y):
        """Divide by a given tensor"""
        result = self.clone()
        if isinstance(y, CrypTensor):
            result._tensor._tensor = torch.broadcast_tensors(
                result._tensor._tensor, y._tensor._tensor
            )[0].clone()
        elif torch.is_tensor(y):
            result._tensor._tensor = torch.broadcast_tensors(result._tensor._tensor, y)[
                0
            ].clone()
        return result.div_(y)

    def div_(self, y):
        if isinstance(y, MPCTensor):
            sign_y = y.sign()
            abs_y = y * sign_y
            return self.mul_(abs_y.reciprocal()).mul_(sign_y)
        self._tensor.div_(y)
        return self


OOP_UNARY_FUNCTIONS = {
    "avg_pool2d": Ptype.arithmetic,
    "sum_pool2d": Ptype.arithmetic,
    "softmax": Ptype.arithmetic,
    "take": Ptype.arithmetic,
    "exp": Ptype.arithmetic,
    "log": Ptype.arithmetic,
    "pow": Ptype.arithmetic,
    "reciprocal": Ptype.arithmetic,
    "sqrt": Ptype.arithmetic,
    "square": Ptype.arithmetic,
    "norm": Ptype.arithmetic,
    "mean": Ptype.arithmetic,
    "neg": Ptype.arithmetic,
    "__neg__": Ptype.arithmetic,
    "cos": Ptype.arithmetic,
    "sin": Ptype.arithmetic,
    "invert": Ptype.binary,
    "lshift": Ptype.binary,
    "rshift": Ptype.binary,
    "__invert__": Ptype.binary,
    "__lshift__": Ptype.binary,
    "__rshift__": Ptype.binary,
    "__rand__": Ptype.binary,
    "__rxor__": Ptype.binary,
    "__ror__": Ptype.binary,
}

OOP_BINARY_FUNCTIONS = {
    "add": Ptype.arithmetic,
    "sub": Ptype.arithmetic,
    "mul": Ptype.arithmetic,
    "matmul": Ptype.arithmetic,
    "conv2d": Ptype.arithmetic,
    "conv_transpose2d": Ptype.arithmetic,
    "dot": Ptype.arithmetic,
    "ger": Ptype.arithmetic,
    "XOR": Ptype.binary,
    "AND": Ptype.binary,
    "OR": Ptype.binary,
    "__xor__": Ptype.binary,
    "__or__": Ptype.binary,
    "__and__": Ptype.binary,
}

INPLACE_UNARY_FUNCTIONS = {
    "neg_": Ptype.arithmetic,
    "invert_": Ptype.binary,
    "lshift_": Ptype.binary,
    "rshift_": Ptype.binary,
}

INPLACE_BINARY_FUNCTIONS = {
    "add_": Ptype.arithmetic,
    "sub_": Ptype.arithmetic,
    "mul_": Ptype.arithmetic,
    "XOR_": Ptype.binary,
    "AND_": Ptype.binary,
    "OR_": Ptype.binary,
    "__ixor__": Ptype.binary,
    "__iand__": Ptype.binary,
}


def _add_oop_unary_passthrough_function(name, preferred=None):
    def ou_wrapper_function(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, name)(*args, **kwargs)
        return result

    if preferred is None:
        setattr(MPCTensor, name, ou_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, False)(ou_wrapper_function))


def _add_oop_binary_passthrough_function(name, preferred=None):
    def ob_wrapper_function(self, value, *args, **kwargs):
        result = self.shallow_copy()
        if isinstance(value, CrypTensor):
            value = value._tensor
        result._tensor = getattr(result._tensor, name)(value, *args, **kwargs)
        return result

    if preferred is None:
        setattr(MPCTensor, name, ob_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, False)(ob_wrapper_function))


def _add_inplace_unary_passthrough_function(name, preferred=None):
    def iu_wrapper_function(self, *args, **kwargs):
        self._tensor = getattr(self._tensor, name)(*args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, name, iu_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, True)(iu_wrapper_function))


def _add_inplace_binary_passthrough_function(name, preferred=None):
    def ib_wrapper_function(self, value, *args, **kwargs):
        if isinstance(value, CrypTensor):
            value = value._tensor
        self._tensor = getattr(self._tensor, name)(value, *args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, name, ib_wrapper_function)
    else:
        setattr(MPCTensor, name, mode(preferred, True)(ib_wrapper_function))


for func_name, preferred_type in OOP_UNARY_FUNCTIONS.items():
    _add_oop_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in OOP_BINARY_FUNCTIONS.items():
    _add_oop_binary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_UNARY_FUNCTIONS.items():
    _add_inplace_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_BINARY_FUNCTIONS.items():
    _add_inplace_binary_passthrough_function(func_name, preferred_type)


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
    "trace",
    "sum",
    "cumsum",
    "reshape",
    "gather",
    "index_select",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, function_name)(*args, **kwargs)
        return result

    setattr(MPCTensor, function_name, regular_func)


def _add_property_function(function_name):
    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    setattr(MPCTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
