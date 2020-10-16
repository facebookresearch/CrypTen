#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import wraps

import crypten
import torch
from crypten import communicator as comm
from crypten.common.tensor_types import is_tensor
from crypten.common.util import (
    ConfigBase,
    adaptive_pool2d_helper,
    pool_reshape,
    torch_cat,
    torch_stack,
)
from crypten.cuda import CUDALongTensor

from ..cryptensor import CrypTensor
from ..encoder import FixedPointEncoder
from .max_helper import _argmax_helper, _max_helper_all_tree_reductions
from .primitives.binary import BinarySharedTensor
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


def _one_hot_to_index(tensor, dim, keepdim, device=None):
    """
    Converts a one-hot tensor output from an argmax / argmin function to a
    tensor containing indices from the input tensor from which the result of the
    argmax / argmin was obtained.
    """
    if dim is None:
        result = tensor.flatten()
        result = result * torch.tensor(list(range(tensor.nelement())), device=device)
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        result = tensor * torch.tensor(
            list(range(tensor.size(dim))), device=device
        ).view(size)
        return result.sum(dim, keepdim=keepdim)


@dataclass
class MPCConfig:
    """
    A configuration object for use by the MPCTensor.
    """

    # exponential function
    exp_iterations: int = 8

    # reciprocal configuration
    reciprocal_method: str = "NR"
    reciprocal_nr_iters: int = 10
    reciprocal_log_iters: int = 1
    reciprocal_all_pos: bool = False
    reciprocal_initial: any = None

    # sigmoid / tanh configuration
    sigmoid_tanh_method: str = "reciprocal"
    sigmoid_tanh_terms: int = 32
    sigmoid_tanh_clip_value: int = 1

    # log configuration
    log_iterations: int = 2
    log_exp_iterations: int = 8
    log_order: int = 8

    # _eix configuration
    _eix_iterations: int = 10

    # Used by max / argmax / min / argmin
    max_method: str = "log_reduction"


# Global config
config = MPCConfig()


class ConfigManager(ConfigBase):
    r"""
    Use this to temporarily change a value in the `mpc.config` object. The
    following sets `config.exp_iterations` to `10` for one function
    invocation and then sets it back to the previous value::

        with ConfigManager("exp_iterations", 10):
            tensor.exp()

    """

    def __init__(self, *args):
        super().__init__(config, *args)


class MPCTensor(CrypTensor):
    def __init__(self, tensor, ptype=Ptype.arithmetic, device=None, *args, **kwargs):
        if tensor is None:
            raise ValueError("Cannot initialize tensor with None.")

        # take required_grad from kwargs, input tensor, or set to False:
        default = tensor.requires_grad if torch.is_tensor(tensor) else False
        requires_grad = kwargs.pop("requires_grad", default)

        # call CrypTensor constructor:
        super().__init__(requires_grad=requires_grad)
        if device is None and hasattr(tensor, "device"):
            device = tensor.device

        # create the MPCTensor:
        tensor_type = ptype.to_tensor()
        if tensor is []:
            self._tensor = torch.tensor([], device=device)
        else:
            self._tensor = tensor_type(tensor, device=device, *args, **kwargs)
        self.ptype = ptype

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, src=0, ptype=Ptype.arithmetic):
        result = MPCTensor([])
        from_shares = ptype.to_tensor().from_shares
        result._tensor = from_shares(share, precision=precision, src=src)
        result.ptype = ptype
        return result

    def clone(self):
        """Create a deep copy of the input tensor."""
        # TODO: Rename this to __deepcopy__()?
        result = MPCTensor([])
        result._tensor = self._tensor.clone()
        result.ptype = self.ptype
        return result

    def shallow_copy(self):
        """Create a shallow copy of the input tensor."""
        # TODO: Rename this to __copy__()?
        result = MPCTensor([])
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    def copy_(self, other):
        """Copies value of other MPCTensor into this MPCTensor."""
        assert isinstance(other, MPCTensor), "other must be MPCTensor"
        self._tensor.copy_(other._tensor)
        self.ptype = other.ptype

    def to(self, *args, **kwargs):
        r"""
        Depending on the input arguments,
        converts underlying share to the given ptype or
        performs `torch.to` on the underlying torch tensor

        To convert underlying share to the given ptype, call `to` as:
            to(ptype, **kwargs)

        It will call MPCTensor.to_ptype with the arguments provided above.

        Otherwise, `to` performs `torch.to` on the underlying
        torch tensor. See
        https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to
        for a reference of the parameters that can be passed in.

        Args:
            ptype: Ptype.arithmetic or Ptype.binary.
        """
        if "ptype" in kwargs:
            return self._to_ptype(**kwargs)
        elif args and isinstance(args[0], Ptype):
            ptype = args[0]
            return self._to_ptype(ptype, **kwargs)
        else:
            share = self.share.to(*args, **kwargs)
            if share.is_cuda:
                share = CUDALongTensor(share)
            self.share = share
            return self

    def _to_ptype(self, ptype, **kwargs):
        r"""
        Convert MPCTensor's underlying share to the corresponding ptype
        (ArithmeticSharedTensor, BinarySharedTensor)

        Args:
            ptype (Ptype.arithmetic or Ptype.binary): The ptype to convert
                the shares to.
            precision (int, optional): Precision of the fixed point encoder when
                converting a binary share to an arithmetic share. It will be ignored
                if the ptype doesn't match.
            bits (int, optional): If specified, will only preserve the bottom `bits` bits
                of a binary tensor when converting from a binary share to an arithmetic share.
                It will be ignored if the ptype doesn't match.
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

    @property
    def device(self):
        """Return the `torch.device` of the underlying share"""
        return self.share.device

    @property
    def is_cuda(self):
        """Return True if the underlying share is stored on GPU, False otherwise"""
        return self.share.is_cuda

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying share"""
        self.share = CUDALongTensor(self.share.cuda(*args, **kwargs))
        return self

    def cpu(self):
        """Call `torch.Tensor.cpu` on the underlying share"""
        self.share = self.share.cpu()
        return self

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        return self._tensor.get_plain_text(dst=dst)

    def reveal(self, dst=None):
        """Decrypts the tensor without any downscaling."""
        return self._tensor.reveal(dst=dst)

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
        return (
            f"MPCTensor(\n\t_tensor={share}\n"
            f"\tplain_text={plain_text}\n\tptype={ptype}\n)"
        )

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if not isinstance(value, MPCTensor):
            value = MPCTensor(value, ptype=self.ptype, device=self.device)
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

    @staticmethod
    def __cat_stack_helper(op, tensors, *args, **kwargs):
        assert op in ["cat", "stack"], "Unsupported op for helper function"
        assert isinstance(tensors, list), "%s input must be a list" % op
        assert len(tensors) > 0, "expected a non-empty list of MPCTensors"

        _ptype = kwargs.pop("ptype", None)
        # Populate ptype field
        if _ptype is None:
            for tensor in tensors:
                if isinstance(tensor, MPCTensor):
                    _ptype = tensor.ptype
                    break
        if _ptype is None:
            _ptype = Ptype.arithmetic

        # Make all inputs MPCTensors of given ptype
        for i, tensor in enumerate(tensors):
            if tensor.ptype != _ptype:
                tensors[i] = tensor.to(_ptype)

        # Operate on all input tensors
        result = tensors[0].clone()
        funcs = {"cat": torch_cat, "stack": torch_stack}
        result.share = funcs[op]([tensor.share for tensor in tensors], *args, **kwargs)
        return result

    @staticmethod
    def cat(tensors, *args, **kwargs):
        """Perform matrix concatenation"""
        return MPCTensor.__cat_stack_helper("cat", tensors, *args, **kwargs)

    @staticmethod
    def stack(tensors, *args, **kwargs):
        """Perform tensor stacking"""
        return MPCTensor.__cat_stack_helper("stack", tensors, *args, **kwargs)

    @staticmethod
    def rand(*sizes, device=None):
        """
        Returns a tensor with elements uniformly sampled in [0, 1). The uniform
        random samples are generated by generating random bits using fixed-point
        encoding and converting the result to an ArithmeticSharedTensor.
        """
        rand = MPCTensor([])
        encoder = FixedPointEncoder()
        rand._tensor = BinarySharedTensor.rand(*sizes, bits=encoder._precision_bits)
        rand._tensor.encoder = encoder
        rand.ptype = Ptype.binary
        return rand.to(Ptype.arithmetic, bits=encoder._precision_bits)

    @staticmethod
    def randn(*sizes, device=None):
        """
        Returns a tensor with normally distributed elements. Samples are
        generated using the Box-Muller transform with optimizations for
        numerical precision and MPC efficiency.
        """
        u = MPCTensor.rand(*sizes).flatten()
        odd_numel = u.numel() % 2 == 1
        if odd_numel:
            u = MPCTensor.cat([u, MPCTensor.rand((1,))])

        n = u.numel() // 2
        u1 = u[:n]
        u2 = u[n:]

        # Radius = sqrt(- 2 * log(u1))
        def sqrtNR(x):
            """
            Newton Raphson method for square root accurate in the range [0, 30]
            which is enough for the full range of log(u) as computed above.

            https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
            """
            # Initialize using efficient polynomial
            y = (1 - (x - 0.5).div(32)).square().square().square() + 0.2

            # Newton Raphson iterations for inverse square root
            for _ in range(3):
                y = y.mul_(3 - x * y.square()).div_(2)

            # Multiply by input to get square root.
            return y * x

        # ln(u) = ln(100u) - ln(100) but log(100u) gives better accuracy in our domain
        r2 = -2 * (u1.mul(100).log() - 4.605170)
        r = sqrtNR(r2)

        # Theta = cos(2 * pi * u2) or sin(2 * pi * u2)
        cos, sin = u2.sub(0.5).mul(6.28318531).cossin()

        # Generating 2 independent normal random variables using
        x = r.mul(sin)
        y = r.mul(cos)
        z = MPCTensor.cat([x, y])

        if odd_numel:
            z = z[1:]

        return z.view(*sizes)

    def bernoulli(self):
        """Returns a tensor with elements in {0, 1}. The i-th element of the
        output will be 1 with probability according to the i-th value of the
        input tensor."""
        return self > MPCTensor.rand(self.size(), device=self.device)

    # TODO: It seems we can remove all Dropout implementations below?
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
        if not training:
            if inplace:
                return self
            else:
                return self.clone()
        rand_tensor = MPCTensor.rand(self.size(), device=self.device)
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
        if not training:
            if inplace:
                return self
            else:
                return self.clone()
        # take first 2 dimensions
        feature_dropout_size = self.size()[0:2]
        # create dropout tensor over the first two dimensions
        rand_tensor = MPCTensor.rand(feature_dropout_size, device=self.device)
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
        if comm.get().get_world_size() == 2:
            return (self - y)._eqz_2PC(_scale=_scale)

        return 1 - self.ne(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def ne(self, y, _scale=True):
        """Returns self != y"""
        if comm.get().get_world_size() == 2:
            return 1 - self.eq(y, _scale=_scale)

        difference = self - y
        difference.share = torch_stack([difference.share, -(difference.share)])
        return difference._ltz(_scale=_scale).sum(0)

    @mode(Ptype.arithmetic)
    def _eqz_2PC(self, _scale=True):
        """Returns self == 0"""
        # Create BinarySharedTensors from shares
        x0 = MPCTensor(self.share, src=0, ptype=Ptype.binary)
        x1 = MPCTensor(-self.share, src=1, ptype=Ptype.binary)

        # Perform equality testing using binary shares
        x0._tensor = x0._tensor.eq(x1._tensor)
        x0.encoder = x0.encoder if _scale else self.encoder

        # Convert to Arithmetic sharing
        result = x0.to(Ptype.arithmetic, bits=1)

        if not _scale:
            result.encoder._scale = 1

        return result

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
        max_weight = x.index_select(
            dim, torch.tensor(x.size(dim) - 1, device=self.device)
        )
        r = MPCTensor.rand(max_weight.size(), device=self.device) * max_weight

        gt = x.gt(r, _scale=False)
        shifted = gt.roll(1, dims=dim)
        shifted.share.index_fill_(dim, torch.tensor(0, device=self.device), 0)

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
    @mode(Ptype.arithmetic)
    def argmax(self, dim=None, keepdim=False, one_hot=True):
        """Returns the indices of the maximum value of all elements in the
        `input` tensor.
        """
        # TODO: Make dim an arg.
        if self.dim() == 0:
            result = (
                MPCTensor(torch.ones((), device=self.device))
                if one_hot
                else MPCTensor(torch.zeros((), device=self.device))
            )
            return result

        result = _argmax_helper(
            self, dim, one_hot, config.max_method, _return_max=False
        )

        if not one_hot:
            result = _one_hot_to_index(result, dim, keepdim, self.device)
        return result

    @mode(Ptype.arithmetic)
    def argmin(self, dim=None, keepdim=False, one_hot=True):
        """Returns the indices of the minimum value of all elements in the
        `input` tensor.
        """
        # TODO: Make dim an arg.
        return (-self).argmax(dim=dim, keepdim=keepdim, one_hot=one_hot)

    @mode(Ptype.arithmetic)
    def max(self, dim=None, keepdim=False, one_hot=True):
        """Returns the maximum value of all elements in the input tensor."""
        # TODO: Make dim an arg.
        method = config.max_method
        if dim is None:
            if method in ["log_reduction", "double_log_reduction"]:
                # max_result can be obtained directly
                max_result = _max_helper_all_tree_reductions(self, method=method)
            else:
                # max_result needs to be obtained through argmax
                with ConfigManager("max_method", method):
                    argmax_result = self.argmax(one_hot=True)
                max_result = self.mul(argmax_result).sum()
            return max_result
        else:
            argmax_result, max_result = _argmax_helper(
                self, dim=dim, one_hot=True, method=method, _return_max=True
            )
            if max_result is None:
                max_result = (self * argmax_result).sum(dim=dim, keepdim=keepdim)
            if keepdim:
                max_result = (
                    max_result.unsqueeze(dim)
                    if max_result.dim() < self.dim()
                    else max_result
                )
            if one_hot:
                return max_result, argmax_result
            else:
                return (
                    max_result,
                    _one_hot_to_index(argmax_result, dim, keepdim, self.device),
                )

    @mode(Ptype.arithmetic)
    def min(self, dim=None, keepdim=False, one_hot=True):
        """Returns the minimum value of all elements in the input tensor."""
        # TODO: Make dim an arg.
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
            # -2 ** 33 is acceptable since it is lower than the supported range
            # which is -2 ** 32 because multiplication can otherwise fail.
            pad_value=(-(2 ** 33)),
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

    def adaptive_avg_pool2d(self, output_size):
        r"""
        Applies a 2D adaptive average pooling over an input signal composed of
        several input planes.

        See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

        Args:
            output_size: the target output size (single integer or
                double-integer tuple)
        """
        resized_input, args, kwargs = adaptive_pool2d_helper(
            self, output_size, reduction="mean"
        )
        return resized_input.avg_pool2d(*args, **kwargs)

    def adaptive_max_pool2d(self, output_size, return_indices=False):
        r"""Applies a 2D adaptive max pooling over an input signal composed of
        several input planes.

        See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

        Args:
            output_size: the target output size (single integer or
                double-integer tuple)
            return_indices: whether to return pooling indices. Default: ``False``
        """
        resized_input, args, kwargs = adaptive_pool2d_helper(
            self, output_size, reduction="max"
        )
        return resized_input.max_pool2d(*args, **kwargs, return_indices=return_indices)

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or MPCTensor): values selected at indices
                where condition is False.

        Returns: MPCTensor or torch.tensor
        """
        if is_tensor(condition):
            condition = condition.float()
            y_masked = y * (1 - condition)
        else:
            # encrypted tensor must be first operand
            y_masked = (1 - condition) * y

        return self * condition + y_masked

    # Logistic Functions
    @mode(Ptype.arithmetic)
    def sigmoid(self):
        """Computes the sigmoid function using the following definition

        .. math::
            \sigma(x) = (1 + e^{-x})^{-1}

        If a valid method is given, this function will compute sigmoid
            using that method:

        "chebyshev" - computes tanh via Chebyshev approximation with
            truncation and uses the identity:

        .. math::
            \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

        Args:
            terms (int): highest degree of Chebyshev polynomials for tanh
                using Chebyshev approximation. Must be even and at least 6.
        """  # noqa: W605
        method = config.sigmoid_tanh_method
        clip_value = config.sigmoid_tanh_clip_value

        if method == "chebyshev":
            tanh_approx = self.div(2).tanh()
            return tanh_approx.div(2) + 0.5
        elif method == "reciprocal":
            ltz = self._ltz(_scale=False)
            sign = 1 - 2 * ltz

            pos_input = self.mul(sign)
            denominator = pos_input.neg().exp().add(1)
            with ConfigManager(
                "exp_iterations",
                9,
                "reciprocal_nr_iters",
                3,
                "reciprocal_all_pos",
                True,
                "reciprocal_initial",
                0.75,
            ):
                pos_output = denominator.reciprocal()

            # Clip values outside of acceptable range
            if clip_value is not None:
                in_range = pos_output.le(clip_value, _scale=False)
                pos_output = pos_output.where(in_range, clip_value)

            result = pos_output.where(1 - ltz, 1 - pos_output)
            # TODO: Support addition with different encoder scales
            # result = pos_output + ltz - 2 * pos_output * ltz
            return result
        else:
            raise ValueError(f"Unrecognized method {method} for sigmoid")

    @mode(Ptype.arithmetic)
    def tanh(self):
        r"""Computes the hyperbolic tangent function using the identity

        .. math::
            tanh(x) = 2\sigma(2x) - 1

        If a valid method is given, this function will compute tanh using that method:

        "chebyshev" - computes tanh via Chebyshev approximation with truncation.

        .. math::
            tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

        where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
        The approximation is truncated to +/-1 outside [-maxval, maxval].

        Args:
            terms (int): highest degree of Chebyshev polynomials.
                         Must be even and at least 6.
        """
        method = config.sigmoid_tanh_method
        terms = config.sigmoid_tanh_terms
        maxval = config.sigmoid_tanh_clip_value

        if method == "reciprocal":
            return self.mul(2).sigmoid().mul(2).sub(1)
        elif method == "chebyshev":
            coeffs = crypten.common.util.chebyshev_series(torch.tanh, maxval, terms)[
                1::2
            ]
            tanh_polys = self.div(maxval)._chebyshev_polynomials(terms)
            tanh_polys_flipped = (
                tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
            )
            out = tanh_polys_flipped.matmul(coeffs)

            # truncate outside [-maxval, maxval]
            return out._truncate_tanh()
        else:
            raise ValueError(f"Unrecognized method {method} for tanh")

    def _truncate_tanh(self):
        """Truncates `out` to +/- clip_value when self is outside [-clip_value, clip_value]."""
        clip_value = config.sigmoid_tanh_clip_value
        if clip_value is None:
            return self
        too_high, too_low = crypten.stack([self, -self]).gt(clip_value)
        in_range = 1 - too_high - too_low
        return (too_high - too_low) * clip_value + self.mul(in_range)

    @mode(Ptype.arithmetic)
    def softmax(self, dim, **kwargs):
        """Compute the softmax of a tensor's elements along a given dimension"""
        # 0-d case
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.ones_like((self.share)))

        if self.size(dim) == 1:
            return MPCTensor(torch.ones_like(self.share))

        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        with ConfigManager("reciprocal_all_pos", True):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
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
            return MPCTensor(torch.zeros((), device=self.device))

        if self.size(dim) == 1:
            return MPCTensor(torch.zeros_like(self.share))

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
            coeffs = torch.tensor(coeffs, device=self.device)
        assert is_tensor(coeffs) or crypten.is_encrypted_tensor(
            coeffs
        ), "Polynomial coefficients must be a list or tensor"
        assert coeffs.dim() == 1, "Polynomial coefficients must be a 1-D tensor"

        # Handle linear case
        if coeffs.size(0) == 1:
            return self.mul(coeffs)

        # Compute terms of polynomial using exponentially growing tree
        terms = crypten.stack([self, self.square()])
        while terms.size(0) < coeffs.size(0):
            highest_term = terms.index_select(
                0, torch.tensor(terms.size(0) - 1, device=self.device)
            )
            new_terms = getattr(terms, func)(highest_term)
            terms = crypten.cat([terms, new_terms])

        # Resize the coefficients for broadcast
        terms = terms[: coeffs.size(0)]
        for _ in range(terms.dim() - 1):
            coeffs = coeffs.unsqueeze(1)

        # Multiply terms by coefficients and sum
        return terms.mul(coeffs).sum(0)

    # Approximations:
    def exp(self):
        """Approximates the exponential function using a limit approximation:

        .. math::

            exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

        Here we compute exp by choosing n = 2 ** d for some large d equal to
        `iterations`. We then compute (1 + x / n) once and square `d` times.

        Set the number of iterations for the limit approximation with
        config.exp_iterations.
        """  # noqa: W605
        result = 1 + self.div(2 ** config.exp_iterations)
        for _ in range(config.exp_iterations):
            result = result.square()
        return result

    def log(self):
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
        iterations = config.log_iterations
        exp_iterations = config.log_exp_iterations
        order = config.log_order

        term1 = self.div(120)
        term2 = self.mul(2).add(1.0).neg().exp().mul(20)
        y = term1 - term2 + 3.0

        # 8th order Householder iterations
        with ConfigManager("exp_iterations", exp_iterations):
            for _ in range(iterations):
                h = 1 - self * (-y).exp()
                y -= h.polynomial([1 / (i + 1) for i in range(order)])
        return y

    def reciprocal(self):
        """
        Methods:
            'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                    of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                    :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default

            'log' : Computes the reciprocal of the input from the observation that:
                    :math:`x^{-1} = exp(-log(x))`

        Configuration params:
            reciprocal_method (str):  One of 'NR' or 'log'.
            reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                         for the `NR` method
            reciprocal_log_iters (int): determines the number of Householder
                iterations to run when computing logarithms for the `log` method
            reciprocal_all_pos (bool): determines whether all elements of the
                input are known to be positive, which optimizes the step of
                computing the sign of the input.
            reciprocal_initial (tensor): sets the initial value for the
                Newton-Raphson method. By default, this will be set to :math:
                `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
                a fairly large domain

        .. _Newton-Raphson:
            https://en.wikipedia.org/wiki/Newton%27s_method
        """
        method = config.reciprocal_method
        if not config.reciprocal_all_pos:
            sgn = self.sign(_scale=False)
            pos = sgn * self
            with ConfigManager("reciprocal_all_pos", True):
                return sgn * pos.reciprocal()

        if method == "NR":
            if config.reciprocal_initial is None:
                # Initialization to a decent estimate (found by qualitative inspection):
                #                1/x = 3exp(.5 - x) + 0.003
                result = 3 * (0.5 - self).exp() + 0.003
            else:
                result = config.reciprocal_initial
            for _ in range(config.reciprocal_nr_iters):
                if isinstance(result, MPCTensor):
                    result += result - result.square().mul_(self)
                else:
                    result = 2 * result - result * result * self
            return result
        elif method == "log":
            return (-(self.log(iterations=config.reciprocal_log_iters))).exp()
        else:
            raise ValueError(f"Invalid method {method} given for reciprocal function")

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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif is_tensor(y):
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
            return self.reciprocal().pow(-p)
        elif p == -1:
            return self.reciprocal()
        elif p == 0:
            # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
            # This is consistent with PyTorch's pow function.
            return MPCTensor(torch.ones_like(self.share))
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
        self.share.set_(result.share.data)
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

    def _eix(self):
        """Computes e^(i * self) where i is the imaginary unit.
        Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
        """
        iterations = config._eix_iterations

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

    def cos(self):
        """Computes the cosine of the input using cos(x) = Re{exp(i * x)}

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self.cossin()[0]

    def sin(self):
        """Computes the sine of the input using sin(x) = Im{exp(i * x)}

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self.cossin()[1]

    def cossin(self):
        """Computes cosine and sine of input via exp(i * x).

        Args:
            iterations (int): for approximating exp(i * x)
        """
        return self._eix()

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
        public = isinstance(tensor, (int, float)) or is_tensor(tensor)
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
        public = isinstance(other, (int, float)) or is_tensor(other)
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
        if is_tensor(src):
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
        results = tuple(
            MPCTensor(0, ptype=self.ptype, device=self.device)
            for _ in range(len(shares))
        )
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def split(self, split_size, dim=0):
        shares = self.share.split(split_size, dim=dim)
        results = tuple(
            MPCTensor(0, ptype=self.ptype, device=self.device)
            for _ in range(len(shares))
        )
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
        if is_tensor(enc_tensor):
            enc_tensor = MPCTensor(enc_tensor)
        assert isinstance(enc_tensor, MPCTensor), "enc_tensor must be an MPCTensor"
        self.share.set_(enc_tensor.share)
        return self


OOP_UNARY_FUNCTIONS = {
    "avg_pool2d": Ptype.arithmetic,
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
    "conv1d": Ptype.arithmetic,
    "conv2d": Ptype.arithmetic,
    "conv_transpose1d": Ptype.arithmetic,
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
        if isinstance(value, MPCTensor):
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
        if isinstance(value, MPCTensor):
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
    "permute",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    """
    Adds function to `MPCTensor` that is applied directly on the underlying
    `_tensor` attribute, and stores the result in the same attribute.
    """

    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, function_name)(*args, **kwargs)
        return result

    setattr(MPCTensor, function_name, regular_func)


def _add_property_function(function_name):
    """
    Adds function to `MPCTensor` that is applied directly on the underlying
    `_tensor` attribute, and returns the result of that function.
    """

    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    setattr(MPCTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
