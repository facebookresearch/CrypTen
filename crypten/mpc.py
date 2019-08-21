#!/usr/bin/env python3

import torch
from crypten.common import EncryptedTensor, constants
from crypten.common.util import pool_reshape
from crypten.primitives.arithmetic.arithmetic import ArithmeticSharedTensor
from crypten.primitives.binary.binary import BinarySharedTensor
from crypten.primitives.converters import convert


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


# TODO: Implement ptype class like torch.dtype
class MPCTensor(EncryptedTensor):
    def __init__(self, input, ptype=ArithmeticSharedTensor, *args, **kwargs):
        if input is None:
            return
        self._tensor = ptype(input, *args, **kwargs)
        self.ptype = ptype

    def shallow_copy(self):
        """Create a shallow copy of the input tensor"""
        result = MPCTensor(None)
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    def to(self, ptype):
        """Converts self._tensor to the given ptype"""
        retval = self.clone()
        if retval.ptype == ptype:
            return retval
        retval._tensor = convert(self._tensor, ptype)
        retval.ptype = ptype
        return retval

    def arithmetic(self):
        """Converts self._tensor to arithmetic secret sharing"""
        return self.to(ArithmeticSharedTensor)

    def binary(self):
        """Converts self._tensor to binary secret sharing"""
        return self.to(BinarySharedTensor)

    def get_plain_text(self):
        """Decrypt the tensor"""
        return self._tensor.get_plain_text()

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if not isinstance(value, MPCTensor):
            value = MPCTensor(value, ptype=self.ptype)
        self._tensor.__setitem__(index, value._tensor)

    # TODO: Move static functions into EncryptedTensor
    @staticmethod
    def cat(tensors, *args, **kwargs):
        """Perform matrix concatenation"""
        assert isinstance(tensors, list), "cat input must be a list"
        assert len(tensors) > 0, "expected a non-empty list of MPCTensors"
        ptype = None
        for tensor in tensors:
            if isinstance(tensor, MPCTensor):
                ptype = tensor.ptype
                break
        if ptype is None:
            ptype = ArithmeticSharedTensor
        for i, tensor in enumerate(tensors):
            if torch.is_tensor(tensor):
                tensors[i] = MPCTensor(tensor, ptype=ptype)
            assert isinstance(
                tensors[i], MPCTensor
            ), "Cannot cat %s with MPCTensor" % type(tensor)
            assert (
                tensors[i].ptype == ptype
            ), "Cannot cat MPCTensors with different ptypes"

        result = tensors[0].shallow_copy()
        result._tensor = result.ptype.cat(
            [tensor._tensor for tensor in tensors], *args, **kwargs
        )
        return result

    @staticmethod
    def stack(tensors, *args, **kwargs):
        assert isinstance(tensors, list), "stack input must be a list"
        assert len(tensors) > 0, "expected a non-empty list of MPCTensors"
        ptype = None
        for tensor in tensors:
            if isinstance(tensor, MPCTensor):
                ptype = tensor.ptype
                break
        if ptype is None:
            ptype = ArithmeticSharedTensor
        for i, tensor in enumerate(tensors):
            if torch.is_tensor(tensor):
                tensors[i] = MPCTensor(tensor)
            assert isinstance(
                tensors[i], MPCTensor
            ), "Can't stack %s with MPCTensor" % type(tensor)
            assert (
                tensors[i].ptype == ptype
            ), "Cannot cat MPCTensors with different ptypes"

        result = tensors[0].shallow_copy()
        result._tensor = result.ptype.stack(
            [tensor._tensor for tensor in tensors], *args, **kwargs
        )
        return result

    @staticmethod
    def bernoulli(tensor):
        """
        Returns a tensor with elements in {0, 1}. The i-th element of the
        output will be 1 with probability according to the i-th value of the
        input tensor.
        """
        result = MPCTensor(None)
        result.ptype = ArithmeticSharedTensor
        result._tensor = ArithmeticSharedTensor.bernoulli(tensor)
        return result

    @staticmethod
    def rand(*sizes):
        """
        Returns a tensor with elements uniformly sampled in [0, 1) using the
        trusted third party.
        """
        result = MPCTensor(None)
        result.ptype = ArithmeticSharedTensor
        result._tensor = ArithmeticSharedTensor.rand(*sizes)
        return result

    @staticmethod
    def randperm(size):
        """
            Generate an MPCTensor with rows that contain values [1, 2, ... n]
            where `n` is the length of each row (size[-1])
        """
        result = MPCTensor(None)
        result.ptype = ArithmeticSharedTensor
        result._tensor = ArithmeticSharedTensor.randperm(size)
        return result

    # Comparators
    @mode(BinarySharedTensor)
    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        return (self >> constants.BITS - 1).arithmetic()

    @mode(ArithmeticSharedTensor)
    def ge(self, y):
        """Returns self >= y"""
        return 1 - self.lt(y)

    @mode(ArithmeticSharedTensor)
    def gt(self, y):
        """Returns self > y"""
        return (-self + y)._ltz()

    @mode(ArithmeticSharedTensor)
    def le(self, y):
        """Returns self <= y"""
        return 1 - self.gt(y)

    @mode(ArithmeticSharedTensor)
    def lt(self, y):
        """Returns self > y"""
        return (self - y)._ltz()

    @mode(ArithmeticSharedTensor)
    def eq(self, y):
        """Returns self == y"""
        return self.ge(y) - self.gt(y)

    @mode(ArithmeticSharedTensor)
    def ne(self, y):
        """Returns self != y"""
        return 1 - self.eq(y)

    @mode(ArithmeticSharedTensor)
    def sign(self):
        """Computes the sign value of a tensor (0 is considered positive)"""
        return 2 * (self >= 0) - 1

    @mode(ArithmeticSharedTensor)
    def abs(self):
        """Computes the absolute value of a tensor"""
        return self * self.sign()

    @mode(ArithmeticSharedTensor)
    def relu(self):
        """Compute a Rectified Linear function on the input tensor.
        """
        return self * (self > 0).to(ArithmeticSharedTensor)

    # max / min-related functions
    def _argmax_helper(self):
        """Returns 1 for all elements that have the highest value in each row"""
        row_length = self.size(-1) if self.size(-1) > 1 else 2

        # Copy each row (length - 1) times to compare to each other row
        a = self.expand(row_length - 1, *self.size())

        # Generate cyclic permutations for each row
        b = MPCTensor.stack([self.roll(i + 1, dims=-1) for i in range(row_length - 1)])

        # Sum of columns with all 1s will have value equal to (length - 1).
        # Using >= since it requires 1-fewer comparrison than !=
        result = (a >= b).sum(dim=0)
        return result >= (row_length - 1)

    @mode(ArithmeticSharedTensor)
    def argmax(self, dim=None, one_hot_required=True):
        """Returns a one-hot vector with a 1 entry at a maximum value.

        If multiple values are equal to the maximum, it will choose one randomly,
        then ties will be broken (randomly) if one_hot_required is True.
        Otherwise, all indices with maximal inputs will be return a 1.
        """
        if dim is None:
            input = self.flatten()
        else:
            input = self.transpose(dim, -1)

        result = input._argmax_helper()

        # Multiply by a random permutation to give each maximum a random priority
        if one_hot_required:
            result *= MPCTensor.randperm(input.size())
            result = result._argmax_helper()

        if dim is None:
            return result.view(self.size())
        else:
            return result.transpose(dim, -1)

    @mode(ArithmeticSharedTensor)
    def argmin(self, **kwargs):
        """Returns a one-hot vector with a 1 entry at a minimum value. If multiple
        values are equal to the minimum, it will choose one randomly"""
        return (-self).argmax(**kwargs)

    @mode(ArithmeticSharedTensor)
    def max(self, dim=None, **kwargs):
        """Compute the max of a tensor's elements (or along a given dimension)"""
        if dim is None:
            return self.mul(self.argmax(**kwargs)).sum()
        else:
            result = self * self.argmax(dim=dim, **kwargs)
            return result.sum(dim=dim)

    @mode(ArithmeticSharedTensor)
    def min(self, **kwargs):
        """Compute the min of a tensor's elements (or along a given dimension)"""
        return -((-self).max(**kwargs))

    @mode(ArithmeticSharedTensor)
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
    @mode(ArithmeticSharedTensor)
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

    @mode(ArithmeticSharedTensor)
    def tanh(self, reciprocal_method="log"):
        """Computes tanh from the sigmoid function:
            tanh(x) = 2 * sigmoid(2 * x) - 1
        """
        return (self * 2).sigmoid(reciprocal_method=reciprocal_method) * 2 - 1

    @mode(ArithmeticSharedTensor)
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
        if isinstance(y, EncryptedTensor):
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
    "avg_pool2d": ArithmeticSharedTensor,
    "sum_pool2d": ArithmeticSharedTensor,
    "softmax": ArithmeticSharedTensor,
    "exp": ArithmeticSharedTensor,
    "log": ArithmeticSharedTensor,
    "pow": ArithmeticSharedTensor,
    "reciprocal": ArithmeticSharedTensor,
    "sqrt": ArithmeticSharedTensor,
    "square": ArithmeticSharedTensor,
    "norm": ArithmeticSharedTensor,
    "mean": ArithmeticSharedTensor,
    "__neg__": ArithmeticSharedTensor,
    "cos": ArithmeticSharedTensor,
    "sin": ArithmeticSharedTensor,
    "invert": BinarySharedTensor,
    "lshift": BinarySharedTensor,
    "rshift": BinarySharedTensor,
    "__invert__": BinarySharedTensor,
    "__lshift__": BinarySharedTensor,
    "__rshift__": BinarySharedTensor,
    "__rand__": BinarySharedTensor,
    "__rxor__": BinarySharedTensor,
    "__ror__": BinarySharedTensor,
}

OOP_BINARY_FUNCTIONS = {
    "add": ArithmeticSharedTensor,
    "sub": ArithmeticSharedTensor,
    "mul": ArithmeticSharedTensor,
    "matmul": ArithmeticSharedTensor,
    "conv2d": ArithmeticSharedTensor,
    "dot": ArithmeticSharedTensor,
    "ger": ArithmeticSharedTensor,
    "XOR": BinarySharedTensor,
    "AND": BinarySharedTensor,
    "OR": BinarySharedTensor,
    "__xor__": BinarySharedTensor,
    "__or__": BinarySharedTensor,
    "__and__": BinarySharedTensor,
}

INPLACE_UNARY_FUNCTIONS = {
    "neg_": ArithmeticSharedTensor,
    "invert_": BinarySharedTensor,
    "lshift_": BinarySharedTensor,
    "rshift_": BinarySharedTensor,
}

INPLACE_BINARY_FUNCTIONS = {
    "add_": ArithmeticSharedTensor,
    "sub_": ArithmeticSharedTensor,
    "mul_": ArithmeticSharedTensor,
    "XOR_": BinarySharedTensor,
    "AND_": BinarySharedTensor,
    "OR_": BinarySharedTensor,
    "__ixor__": BinarySharedTensor,
    "__iand__": BinarySharedTensor,
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
        if isinstance(value, EncryptedTensor):
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
        if isinstance(value, EncryptedTensor):
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
