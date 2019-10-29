#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch
from crypten.common.util import pool_reshape

from ..autograd_cryptensor import AutogradCrypTensor
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


def _one_hot_to_index(tensor, dim, keepdim):
    """
    Converts a one-hot tensor output from an argmax / argmin function to a
    tensor containing indices from the input tensor from which the result of the
    argmax / argmin was obtained.
    """
    if dim is None:
        result = tensor.flatten()
        result = result * torch.tensor([i for i in range(tensor.nelement())])
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        result = tensor * torch.tensor([i for i in range(tensor.size(dim))]).view(size)
        return result.sum(dim, keepdim=keepdim)


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
        """Converts self._tensor to the given ptype

        Args:
            ptype: Ptype.arithmetic or Ptype.binary.
        """
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
        """Decrypts the tensor"""
        return self._tensor.get_plain_text()

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate MPCTensors to boolean values")

    def __repr__(self):
        """Returns a representation of the tensor useful for debugging."""
        from crypten.debug import debug_mode

        share = self.share
        plain_text = self._tensor.get_plain_text() if debug_mode() else "HIDDEN"
        ptype = self.ptype
        repr = (
            f"MPCTensor(\n\t_tensor={share}\n"
            f"\tplain_text={plain_text}\n\tptype={ptype}\n)"
        )
        return repr

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if not isinstance(value, MPCTensor):
            value = MPCTensor(value, ptype=self.ptype)
        self._tensor.__setitem__(index, value._tensor)

    @property
    def share(self):
        """Returns underlying _tensor"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets _tensor to value"""
        self._tensor.share = value

    def bernoulli(self):
        """Draws a random tensor from {0, 1} with probability 0.5"""
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
        # TODO: Adapt this to take a dim argument.
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
    def argmax(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the maximum value of all elements in the
        `input` tensor.
        """
        if self.dim() == 0:
            return MPCTensor(torch.ones(())) if one_hot else MPCTensor(torch.zeros(()))

        input = self.flatten() if dim is None else self.transpose(dim, -1)

        result = input._argmax_helper()

        # Multiply by a random permutation to give each maximum a random priority
        result *= crypten.mpc.randperm(input.size())
        result = result._argmax_helper()

        result = result.view(self.size()) if dim is None else result.transpose(dim, -1)
        return result if one_hot else _one_hot_to_index(result, dim, keepdim)

    @mode(Ptype.arithmetic)
    def argmin(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the minimum value of all elements in the
        `input` tensor.
        """
        return (-self).argmax(dim=dim, keepdim=keepdim, one_hot=one_hot)

    @mode(Ptype.arithmetic)
    def max(self, dim=None, keepdim=False, one_hot=False):
        """Returns the maximum value of all elements in the input tensor."""
        if dim is None:
            argmax_result = self.argmax(one_hot=True)
            max_result = self.mul(argmax_result).sum()
            return max_result
        else:
            argmax_result = self.argmax(dim=dim, one_hot=True)
            max_result = (self * argmax_result).sum(dim=dim, keepdim=keepdim)
            if one_hot:
                return max_result, argmax_result
            else:
                return max_result, _one_hot_to_index(argmax_result, dim, keepdim)

    @mode(Ptype.arithmetic)
    def min(self, dim=None, keepdim=False, one_hot=False):
        """Returns the minimum value of all elements in the input tensor."""
        result = (-self).max(dim=dim, keepdim=keepdim, one_hot=one_hot)
        if dim is None:
            return -result
        else:
            return -result[0], result[1]

    @mode(Ptype.arithmetic)
    def max_pool2d(self, kernel_size, padding=None, stride=None, return_indices=False):
        """Applies a 2D max pooling over an input signal composed of several
        input planes.
        """
        max_input = self.shallow_copy()
        max_input.share, output_size = pool_reshape(
            self.share,
            kernel_size,
            padding=padding,
            stride=stride,
            # padding with extremely negative values to avoid choosing pads
            # -2 ** 40 is acceptable since it is lower than the supported range
            # which is -2 ** 32 because multiplication can otherwise fail.
            pad_value=(-2 ** 40),
        )
        max_vals, argmax_vals = max_input.max(dim=-1, one_hot=True)
        max_vals = max_vals.view(output_size)
        if return_indices:
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            argmax_vals = argmax_vals.view(output_size + kernel_size)
            return max_vals, argmax_vals
        return max_vals

    @mode(Ptype.arithmetic)
    def _max_pool2d_backward(
        self, indices, kernel_size, padding=None, stride=None, output_size=None
    ):
        """Implements the backwards for a `max_pool2d` call."""
        # Setup padding
        if padding is None:
            padding = 0
        if isinstance(padding, int):
            padding = padding, padding
        assert isinstance(padding, tuple), "padding must be a int, tuple, or None"
        p0, p1 = padding

        # Setup stride
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = stride, stride
        assert isinstance(padding, tuple), "stride must be a int, tuple, or None"
        s0, s1 = stride

        # Setup kernel_size
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        assert isinstance(padding, tuple), "padding must be a int or tuple"
        k0, k1 = kernel_size

        assert self.dim() == 4, "Input to _max_pool2d_backward must have 4 dimensions"
        assert (
            indices.dim() == 6
        ), "Indices input for _max_pool2d_backward must have 6 dimensions"

        # Computes one-hot gradient blocks from each output variable that
        # has non-zero value corresponding to the argmax of the corresponding
        # block of the max_pool2d input.
        kernels = self.view(self.size() + (1, 1)) * indices

        # Use minimal size if output_size is not specified.
        if output_size is None:
            output_size = (
                self.size(0),
                self.size(1),
                s0 * self.size(2) - 2 * p0,
                s1 * self.size(3) - 2 * p1,
            )

        # Sum the one-hot gradient blocks at corresponding index locations.
        result = MPCTensor(torch.zeros(output_size)).pad([p0, p0, p1, p1])
        for i in range(self.size(2)):
            for j in range(self.size(3)):
                left_ind = s0 * i
                top_ind = s1 * j

                result[
                    :, :, left_ind : left_ind + k0, top_ind : top_ind + k1
                ] += kernels[:, :, i, j]

        result = result[:, :, p0 : result.size(2) - p0, p1 : result.size(3) - p1]
        return result

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or MPCTensor): values selected at indices
                where condition is False.

        Returns: MPCTensor or torch.tensor
        """
        if torch.is_tensor(condition):
            condition = condition.float()
            y_masked = y * (1 - condition)
        else:
            # encrypted tensor must be first operand
            y_masked = (1 - condition) * y

        return self * condition + y_masked

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
    def softmax(self, dim, **kwargs):
        """Compute the softmax of a tensor's elements along a given dimension
        """
        # 0-d case
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.ones(()))

        if self.size(dim) == 1:
            return MPCTensor(torch.ones(self.size()))

        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        denominator = numerator.sum(dim, keepdim=True)
        return numerator / denominator

    @mode(Ptype.arithmetic)
    def pad(self, pad, mode="constant", value=0):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            result._tensor = self._tensor.pad(pad, mode=mode, value=value._tensor)
        else:
            result._tensor = self._tensor.pad(pad, mode=mode, value=value)
        return result

    # Approximations:
    def exp(self, iterations=8):
        """Approximates the exponential function using a limit approximation:

        .. math::

            exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

        Here we compute exp by choosing n = 2 ** d for some large d equal to
        `iterations`. We then compute (1 + x / n) once and square `d` times.

        Args:
            iterations (int): number of iterations for limit approximation
        """
        result = 1 + self.div(2 ** iterations)
        for _ in range(iterations):
            result = result.square()
        return result

    def log(self, iterations=2, exp_iterations=8):
        """Approximates the natural logarithm using 6th order modified
        Householder iterations.

        Iterations are computed by: :math:`h = 2 - x * exp(-y_n)`

        .. math::

            y_{n+1} = y_n - h * (1 + h / 2 + h^2 / 3 + h^3 / 6 + h^4 / 5 + h^5 / 7)

        Args:
            iterations (int): number of iterations for 6th order modified
                Householder approximation.
            exp_iterations (int): number of iterations for limit approximation of exp
        """

        # Initialization to a decent estimate (found by qualitative inspection):
        #                ln(x) = x/40 - 8exp(-2x - .3) + 1.9
        term1 = self / 40
        term2 = 8 * (-2 * self - 0.3).exp()
        y = term1 - term2 + 1.9

        # 6th order Householder iterations
        for _ in range(iterations):
            h = 1 - self * (-y).exp(iterations=exp_iterations)
            h2 = h.square()
            h3 = h2 * h
            h4 = h2.square()
            h5 = h4 * h
            y -= h * (1 + h.div(2) + h2.div_(3) + h3.div_(6) + h4.div_(5) + h5.div_(7))

        return y

    def reciprocal(self, method="NR", nr_iters=10, log_iters=1, all_pos=False):
        """
        Methods:
            'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                    of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                    :math:`3*exp(-(x-.5)) + 0.003` as an initial guess

            'log' : Computes the reciprocal of the input from the observation that:
                    :math:`x^{-1} = exp(-log(x))`

        Args:
            nr_iters (int):  determines the number of Newton-Raphson iterations to run
                         for the `NR` method
            log_iters (int): determines the number of Householder iterations to run
                         when computing logarithms for the `log` method
            all_pos (bool): determines whether all elements
                       of the input are known to be positive, which optimizes
                       the step of computing the sign of the input.

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """
        if not all_pos:
            sgn = self.sign()
            abs = sgn * self
            return sgn * abs.reciprocal(
                method=method, nr_iters=nr_iters, log_iters=log_iters, all_pos=True
            )

        if method == "NR":
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(.5 - x) + 0.003
            result = 3 * (0.5 - self).exp() + 0.003
            for _ in range(nr_iters):
                result += result - result.square().mul_(self)
            return result
        elif method == "log":
            return (-self.log(iterations=log_iters)).exp()
        else:
            raise ValueError("Invalid method %s given for reciprocal function" % method)

    def div(self, y):
        r"""Divides each element of :attr:`self` with the scalar :attr:`y` or
        each element of the tensor :attr:`y` and returns a new resulting tensor.

        For `y` a scalar:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}}

        For `y` a tensor:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}_i}

        Note for :attr:`y` a tensor, the shapes of :attr:`self` and :attr:`y` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif torch.is_tensor(y):
            result.share = torch.broadcast_tensors(result.share, y)[0].clone()
        return result.div_(y)

    def div_(self, y):
        """In-place version of :meth:`div`"""
        if isinstance(y, MPCTensor):
            return self.mul_(y.reciprocal())
        self._tensor.div_(y)
        return self

    def pow(self, p, **kwargs):
        """
        Computes an element-wise exponent `p` of a tensor, where `p` is an
        integer.
        """
        # TODO: Make an inplace version to be consistent with PyTorch
        if isinstance(p, float) and int(p) == p:
            p = int(p)

        if not isinstance(p, int):
            raise TypeError(
                "pow must take an integer exponent. For non-integer powers, use"
                " pos_pow with positive-valued base."
            )
        if p < -1:
            return self.reciprocal(**kwargs).pow(-p)
        elif p == -1:
            return self.reciprocal(**kwargs)
        elif p == 0:
            # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
            # This is consistent with PyTorch's pow function.
            return MPCTensor(torch.ones(self.size()))
        elif p == 1:
            return self.clone()
        elif p == 2:
            return self.square()
        elif p % 2 == 0:
            return self.square().pow(p // 2)
        else:
            return self.square().mul_(self).pow((p - 1) // 2)

    def pos_pow(self, p):
        """
        Approximates self ** p by computing: :math:`x^p = exp(p * log(x))`

        Note that this requires that the base `self` contain only positive values
        since log can only be computed on positive numbers.

        Note that the value of `p` can be an integer, float, public tensor, or
        encrypted tensor.
        """
        if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
            return self.pow(p)
        return self.log().mul_(p).exp()

    def sqrt(self):
        """
        Computes the square root of the input by raising it to the 0.5 power
        """
        return self.pos_pow(0.5)

    def norm(self, p="fro", dim=None, keepdim=False):
        """Computes the p-norm of the input tensor (or along a dimension)."""
        if p == "fro":
            p = 2

        if isinstance(p, (int, float)):
            assert p >= 1, "p-norm requires p >= 1"
            if p == 1:
                if dim is None:
                    return self.abs().sum()
                return self.abs().sum(dim, keepdim=keepdim)
            elif p == 2:
                if dim is None:
                    return self.square().sum().sqrt()
                return self.square().sum(dim, keepdim=keepdim).sqrt()
            elif p == float("inf"):
                if dim is None:
                    return self.abs().max()
                return self.abs().max(dim=dim, keepdim=keepdim)[0]
            else:
                if dim is None:
                    return self.abs().pos_pow(p).sum().pos_pow(1 / p)
                return self.abs().pos_pow(p).sum(dim, keepdim=keepdim).pos_pow(1 / p)
        elif p == "nuc":
            raise NotImplementedError("Nuclear norm is not implemented")
        else:
            raise ValueError(f"Improper value p ({p})for p-norm")

    def _eix(self, iterations=10):
        """Computes e^(i * self) where i is the imaginary unit.
        Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
        """
        re = 1
        im = self.div(2 ** iterations)

        # First iteration uses knowledge that `re` is public and = 1
        re -= im.square()
        im *= 2

        # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
        for _ in range(iterations - 1):
            a2 = re.square()
            b2 = im.square()
            im = im.mul_(re)
            im._tensor *= 2
            re = a2 - b2

        return re, im

    def cos(self, iterations=10):
        """Computes the cosine of the input using cos(x) = Re{exp(i * x)}

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self.cossin(iterations=iterations)[0]

    def sin(self, iterations=10):
        """Computes the sine of the input using sin(x) = Im{exp(i * x)}

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self.cossin(iterations=iterations)[1]

    def cossin(self, iterations=10):
        """Computes cosine and sine of input via exp(i * x).

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self._eix(iterations=iterations)

    def index_add(self, dim, index, tensor):
        """Performs out-of-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index.
        """
        return self.clone().index_add_(dim, index, tensor)

    def index_add_(self, dim, index, tensor):
        """Performs in-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index.
        """
        assert index.dim() == 1, "index needs to be a vector"
        public = isinstance(tensor, (int, float)) or torch.is_tensor(tensor)
        private = isinstance(tensor, MPCTensor)
        if public:
            self._tensor.index_add_(dim, index, tensor)
        elif private:
            self._tensor.index_add_(dim, index, tensor._tensor)
        else:
            raise TypeError("index_add second tensor of unsupported type")
        return self

    def scatter_add(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor.
        """
        return self.clone().scatter_add_(dim, index, other)

    def scatter_add_(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor."""
        public = isinstance(other, (int, float)) or torch.is_tensor(other)
        private = isinstance(other, CrypTensor)
        if public:
            self._tensor.scatter_add_(dim, index, other)
        elif private:
            self._tensor.scatter_add_(dim, index, other._tensor)
        else:
            raise TypeError("scatter_add second tensor of unsupported type")
        return self

    def scatter_(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        if torch.is_tensor(src):
            src = MPCTensor(src)
        assert isinstance(src, MPCTensor), "Unrecognized scatter src type: %s" % type(
            src
        )
        self.share.scatter_(dim, index, src.share)
        return self

    def scatter(self, dim, index, src):
        """Out-of-place version of :meth:`MPCTensor.scatter_`"""
        result = self.clone()
        return result.scatter_(dim, index, src)

    def unbind(self, dim=0):
        shares = self.share.unbind(dim=dim)
        results = tuple(MPCTensor(0, ptype=self.ptype) for _ in range(len(shares)))
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def split(self, split_size, dim=0):
        shares = self.share.split(split_size, dim=dim)
        results = tuple(MPCTensor(0, ptype=self.ptype) for _ in range(len(shares)))
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def set(self, enc_tensor):
        """
        Sets self encrypted to enc_tensor in place by setting
        shares of self to those of enc_tensor.

        Args:
            enc_tensor (MPCTensor): with encrypted shares.
        """
        if torch.is_tensor(enc_tensor):
            enc_tensor = MPCTensor(enc_tensor)
        assert isinstance(enc_tensor, MPCTensor), "enc_tensor must be an MPCTensor"
        self.share.set_(enc_tensor.share)
        return self


OOP_UNARY_FUNCTIONS = {
    "avg_pool2d": Ptype.arithmetic,
    "sum_pool2d": Ptype.arithmetic,
    "take": Ptype.arithmetic,
    "square": Ptype.arithmetic,
    "mean": Ptype.arithmetic,
    "var": Ptype.arithmetic,
    "neg": Ptype.arithmetic,
    "__neg__": Ptype.arithmetic,
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
    "__ior__": Ptype.binary,
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
        if isinstance(value, AutogradCrypTensor):
            value = value._tensor
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
        if isinstance(value, AutogradCrypTensor):
            value = value._tensor
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
