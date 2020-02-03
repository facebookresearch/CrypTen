#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from functools import wraps

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
            # @wraps ensures docstrings are updated
            @wraps(func)
            def convert_wrapper(self, *args, **kwargs):
                self._tensor = convert(self._tensor, ptype)
                self.ptype = ptype
                self = func(self, *args, **kwargs)
                return self

            return convert_wrapper

    else:

        def function_wrapper(func):
            # @wraps ensures docstrings are updated
            @wraps(func)
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

    def get_plain_text(self, dst=None):
        """Decrypts the tensor"""
        return self._tensor.get_plain_text(dst=dst)

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
        """Returns underlying share"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets share to value"""
        self._tensor.share = value

    @property
    def encoder(self):
        """Returns underlying encoder"""
        return self._tensor.encoder

    @encoder.setter
    def encoder(self, value):
        """Sets encoder to value"""
        self._tensor.encoder = value

    def bernoulli(self):
        """Returns a tensor with elements in {0, 1}. The i-th element of the
        output will be 1 with probability according to the i-th value of the
        input tensor."""
        return self > crypten.mpc.rand(self.size())

    def dropout(self, p=0.5, training=True, inplace=False):
        r"""
        Randomly zeroes some of the elements of the input tensor with
        probability :attr:`p`.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place.
                Default: ``False``
        """
        assert p >= 0.0 and p <= 1.0, "dropout probability has to be between 0 and 1"
        if training is False:
            if inplace:
                return self
            else:
                return self.clone()
        rand_tensor = crypten.mpc.rand(self.size())
        dropout_tensor = rand_tensor > p
        if inplace:
            result_tensor = self.mul_(dropout_tensor).div_(1 - p)
        else:
            result_tensor = self.mul(dropout_tensor).div_(1 - p)
        return result_tensor

    def dropout2d(self, p=0.5, training=True, inplace=False):
        r"""
        Randomly zero out entire channels (a channel is a 2D feature map,
        e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
        batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
        Each channel will be zeroed out independently on every forward call with
        probability :attr:`p` using samples from a Bernoulli distribution.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place.
                Default: ``False``
        """
        assert p >= 0.0 and p <= 1.0, "dropout probability has to be between 0 and 1"
        return self._feature_dropout(p, training, inplace)

    def dropout3d(self, p=0.5, training=True, inplace=False):
        r"""
        Randomly zero out entire channels (a channel is a 3D feature map,
        e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
        batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
        Each channel will be zeroed out independently on every forward call with
        probability :attr:`p` using samples from a Bernoulli distribution.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place.
                Default: ``False``
        """
        # This is 100% the same code as dropout2d. We duplicate this code so that
        # stack traces are not confusing.
        assert p >= 0.0 and p <= 1.0, "dropout probability has to be between 0 and 1"
        return self._feature_dropout(p, training, inplace)

    def _feature_dropout(self, p=0.5, training=True, inplace=False):
        """Randomly zeros out entire channels in the input tensor with probability
        :attr:`p`. (a channel is a nD feature map, e.g., the :math:`j`-th channel
        of the :math:`i`-th sample in the batched input is a nD tensor
        :math:`\text{input}[i, j]`)."""
        assert self.dim() >= 2, "feature dropout requires dimension to be at least 2"
        assert p >= 0.0 and p <= 1.0, "dropout probability has to be between 0 and 1"
        if training is False:
            if inplace:
                return self
            else:
                return self.clone()
        # take first 2 dimensions
        feature_dropout_size = self.size()[0:2]
        # create dropout tensor over the first two dimensions
        rand_tensor = crypten.mpc.rand(feature_dropout_size)
        feature_dropout_tensor = rand_tensor > p
        # Broadcast to remaining dimensions
        for i in range(2, self.dim()):
            feature_dropout_tensor = feature_dropout_tensor.unsqueeze(i)
        feature_dropout_tensor.share, self.share = torch.broadcast_tensors(
            feature_dropout_tensor.share, self.share
        )
        if inplace:
            result_tensor = self.mul_(feature_dropout_tensor).div_(1 - p)
        else:
            result_tensor = self.mul(feature_dropout_tensor).div_(1 - p)
        return result_tensor

    # Comparators
    @mode(Ptype.binary)
    def _ltz(self, _scale=True):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        result = (self >> shift).to(Ptype.arithmetic, bits=1)
        if _scale:
            return result * result.encoder._scale
        else:
            result.encoder._scale = 1
            return result

    @mode(Ptype.arithmetic)
    def ge(self, y, _scale=True):
        """Returns self >= y"""
        return 1 - self.lt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def gt(self, y, _scale=True):
        """Returns self > y"""
        return (-self + y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def le(self, y, _scale=True):
        """Returns self <= y"""
        return 1 - self.gt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def lt(self, y, _scale=True):
        """Returns self < y"""
        return (self - y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def eq(self, y, _scale=True):
        """Returns self == y"""
        return 1 - self.ne(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def ne(self, y, _scale=True):
        """Returns self != y"""
        difference = self - y
        difference.share = torch.stack([difference.share, -(difference.share)])
        return difference._ltz(_scale=_scale).sum(0)

    @mode(Ptype.arithmetic)
    def sign(self, _scale=True):
        """Computes the sign value of a tensor (0 is considered positive)"""
        return 1 - 2 * self._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def abs(self):
        """Computes the absolute value of a tensor"""
        return self * self.sign(_scale=False)

    @mode(Ptype.arithmetic)
    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        return self * self.ge(0, _scale=False)

    @mode(Ptype.arithmetic)
    def weighted_index(self, dim=None):
        """
        Returns a tensor with entries that are one-hot along dimension `dim`.
        These one-hot entries are set at random with weights given by the input
        `self`.

        Examples::

            >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
            >>> index = encrypted_tensor.weighted_index().get_plain_text()
            # With 1 / 7 probability
            torch.tensor([1., 0.])

            # With 6 / 7 probability
            torch.tensor([0., 1.])
        """
        if dim is None:
            return self.flatten().weighted_index(dim=0).view(self.size())

        x = self.cumsum(dim)
        max_weight = x.index_select(dim, torch.tensor(x.size(dim) - 1))
        r = crypten.mpc.rand(max_weight.size()) * max_weight

        gt = x.gt(r, _scale=False)
        shifted = gt.roll(1, dims=dim)
        shifted.share.index_fill_(dim, torch.tensor(0), 0)

        return gt - shifted

    @mode(Ptype.arithmetic)
    def weighted_sample(self, dim=None):
        """
        Samples a single value across dimension `dim` with weights corresponding
        to the values in `self`

        Returns the sample and the one-hot index of the sample.

        Examples::

            >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
            >>> index = encrypted_tensor.weighted_sample().get_plain_text()
            # With 1 / 7 probability
            (torch.tensor([1., 0.]), torch.tensor([1., 0.]))

            # With 6 / 7 probability
            (torch.tensor([0., 6.]), torch.tensor([0., 1.]))
        """
        indices = self.weighted_index(dim)
        sample = self.mul(indices).sum(dim)
        return sample, indices

    # max / min-related functions
    def _argmax_helper(self, dim=None):
        """Returns 1 for all elements that have the highest value in the appropriate
           dimension of the tensor.
        """

        dim = -1 if dim is None else dim
        row_length = self.size(dim) if self.size(dim) > 1 else 2

        # Copy each row (length - 1) times to compare to each other row
        a = self.expand(row_length - 1, *self.size())

        # Generate cyclic permutations for each row
        b = crypten.mpc.stack(
            [self.roll(i + 1, dims=dim) for i in range(row_length - 1)]
        )

        # Use either prod or sum & comparison depending on size
        if row_length - 1 < torch.iinfo(torch.long).bits * 2:
            pairwise_comparisons = a.ge(b, _scale=False)
            result = pairwise_comparisons.prod(dim=0)
            result.share *= self.encoder._scale
            result.encoder = self.encoder
        else:
            # Sum of columns with all 1s will have value equal to (length - 1).
            # Using ge() since it is slightly faster than eq()
            pairwise_comparisons = a.ge(b)
            result = pairwise_comparisons.sum(dim=0).ge(row_length - 1)
        return result

        """
        pairwise_comparisons = a.ge(b, _scale=False)

        return result
        """

    @mode(Ptype.arithmetic)
    def argmax(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the maximum value of all elements in the
        `input` tensor.
        """
        if self.dim() == 0:
            return MPCTensor(torch.ones(())) if one_hot else MPCTensor(torch.zeros(()))

        input = self.flatten() if dim is None else self
        result = input._argmax_helper(dim)

        # Break ties by using a uniform weighted sample among tied indices
        result = result.weighted_index(dim)

        result = result.view(self.size()) if dim is None else result
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
    def sigmoid(self, maxval_tanh=6, terms_tanh=32, reciprocal_method=None):
        """Computes the sigmoid function as
                sigmoid(x) = (tanh(x /2) + 1) / 2
        Args:
            maxval_tanh (int): interval width used for tanh chebyshev polynomials
            terms_tanh (int): highest degree of Chebyshev polynomials for tanh.
                         Must be even and at least 6.
        """
        if reciprocal_method:
            warnings.warn(
                "reciprocal_method is deprecated in favor of Chebyshev approximations",
                DeprecationWarning,
            )

        tanh_approx = self.div(2).tanh(maxval=maxval_tanh, terms=terms_tanh)
        return tanh_approx.div(2) + 0.5

    @mode(Ptype.arithmetic)
    def tanh(self, maxval=6, terms=32, reciprocal_method=None):
        r"""Computes tanh via Chebyshev approximation with truncation.

        .. math::
            tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

        where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
        The approximation is truncated to +/-1 outside [-maxval, maxval].

        Args:
            maxval (int): interval width used for computing chebyshev polynomials
            terms (int): highest degree of Chebyshev polynomials.
                         Must be even and at least 6.
        """
        if reciprocal_method:
            warnings.warn(
                "reciprocal_method is deprecated in favor of Chebyshev approximations",
                DeprecationWarning,
            )

        coeffs = crypten.common.util.chebyshev_series(torch.tanh, maxval, terms)[1::2]
        tanh_polys = self.div(maxval)._chebyshev_polynomials(terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)
        # truncate outside [-maxval, maxval]
        out = self._truncate_tanh(maxval, out)
        return out

    def _truncate_tanh(self, maxval, out):
        """Truncates `out` to +/-1 when self is outside [-maxval, maxval].

        Args:
            maxval (int): interval width outside of which to truncate
            out (torch.tensor or MPCTensor): tensor to truncate
        """
        too_high, too_low = crypten.stack([self, -self]).gt(maxval)
        in_range = -too_high - too_low + 1
        out = too_high - too_low + out.mul(in_range)
        return out

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
        inv_denominator = numerator.sum(dim, keepdim=True).reciprocal(all_pos=True)
        return numerator * inv_denominator

    @mode(Ptype.arithmetic)
    def log_softmax(self, dim, **kwargs):
        """Applies a softmax followed by a logarithm.
        While mathematically equivalent to log(softmax(x)), doing these two
        operations separately is slower, and numerically unstable. This function
        uses an alternative formulation to compute the output and gradient correctly.
        """
        # 0-d case
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.zeros(()))

        if self.size(dim) == 1:
            return MPCTensor(torch.zeros(self.size()))

        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        normalize_term = logits.exp().sum(dim, keepdim=True)
        result = logits - normalize_term.log()
        return result

    @mode(Ptype.arithmetic)
    def pad(self, pad, mode="constant", value=0):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            result._tensor = self._tensor.pad(pad, mode=mode, value=value._tensor)
        else:
            result._tensor = self._tensor.pad(pad, mode=mode, value=value)
        return result

    @mode(Ptype.arithmetic)
    def polynomial(self, coeffs, func="mul"):
        """Computes a polynomial function on a tensor with given coefficients,
        `coeffs`, that can be a list of values or a 1-D tensor.

        Coefficients should be ordered from the order 1 (linear) term first,
        ending with the highest order term. (Constant is not included).
        """
        # Coefficient input type-checking
        if isinstance(coeffs, list):
            coeffs = torch.tensor(coeffs)
        assert torch.is_tensor(coeffs) or crypten.is_encrypted_tensor(
            coeffs
        ), "Polynomial coefficients must be a list or tensor"
        assert coeffs.dim() == 1, "Polynomial coefficients must be a 1-D tensor"

        # Handle linear case
        if coeffs.size(0) == 1:
            return self.mul(coeffs)

        # Compute terms of polynomial using exponentially growing tree
        terms = crypten.mpc.stack([self, self.square()])
        while terms.size(0) < coeffs.size(0):
            highest_term = terms[-1:].expand(terms.size())
            new_terms = getattr(terms, func)(highest_term)
            terms = crypten.cat([terms, new_terms])

        # Resize the coefficients for broadcast
        terms = terms[: coeffs.size(0)]
        for _ in range(terms.dim() - 1):
            coeffs = coeffs.unsqueeze(1)

        # Multiply terms by coefficients and sum
        return terms.mul(coeffs).sum(0)

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

    def log(self, iterations=2, exp_iterations=8, order=8):
        r"""
        Approximates the natural logarithm using 8th order modified
        Householder iterations. This approximation is accurate within 2% relative
        error on [0.0001, 250].

        Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

        .. math::

            y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

        Args:
            iterations (int): number of Householder iterations for the approximation
            exp_iterations (int): number of iterations for limit approximation of exp
            order (int): number of polynomial terms used (order of Householder approx)
        """

        # Initialization to a decent estimate (found by qualitative inspection):
        #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
        term1 = self.div(120)
        term2 = self.mul(2).add(1.0).neg().exp().mul(20)
        y = term1 - term2 + 3.0

        # 8th order Householder iterations
        for _ in range(iterations):
            h = 1 - self * (-y).exp(iterations=exp_iterations)
            y -= h.polynomial([1 / (i + 1) for i in range(order)])
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
            sgn = self.sign(_scale=False)
            abs = sgn * self
            rec = abs.reciprocal(
                method=method, nr_iters=nr_iters, log_iters=log_iters, all_pos=True
            )
            return sgn * rec

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

    def pow_(self, p, **kwargs):
        """In-place version of pow_ function"""
        result = self.pow(p)
        self.share.data = result.share.data
        return self

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

    def _chebyshev_polynomials(self, terms):
        r"""Evaluates odd degree Chebyshev polynomials at x

        Chebyshev Polynomials of the first kind are defined as

        .. math::
            P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

        Args:
            self (MPCTensor): input at which polynomials are evaluated
            terms (int): highest degree of Chebyshev polynomials.
                         Must be even and at least 6.
        Returns:
            MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
        """
        if terms % 2 != 0 or terms < 6:
            raise ValueError("Chebyshev terms must be even and >= 6")

        polynomials = [self.clone()]
        y = 4 * self.square() - 2
        z = y - 1
        polynomials.append(z.mul(self))

        for k in range(2, terms // 2):
            next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
            polynomials.append(next_polynomial)

        return crypten.stack(polynomials)

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
    "prod",
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
