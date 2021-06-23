#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import wraps

import torch
from crypten import communicator as comm
from crypten.common.tensor_types import is_tensor
from crypten.common.util import (
    ConfigBase,
    torch_cat,
    torch_stack,
)
from crypten.cuda import CUDALongTensor

from ..cryptensor import CrypTensor
from ..encoder import FixedPointEncoder
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


@dataclass
class MPCConfig:
    """
    A configuration object for use by the MPCTensor.
    """

    # Used by max / argmax / min / argmin
    max_method: str = "log_reduction"

    # Used (for the moment) when generating the Beaver Triples
    active_security: bool = False


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
        """
        Creates the shared tensor from the input `tensor` provided by party `src`.
        The `ptype` defines the type of sharing used (default: arithmetic).

        The other parties can specify a `tensor` or `size` to determine the size
        of the shared tensor object to create. In this case, all parties must
        specify the same (tensor) size to prevent the party's shares from varying
        in size, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct size. The parties who do not know the
        tensor size beforehand can provide an empty tensor as input. This is
        guaranteed to produce correct behavior but requires an additional
        communication round.

        The parties can also set the `precision` and `device` for their share of
        the tensor. If `device` is unspecified, it is set to `tensor.device`.
        """
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
            self._tensor = tensor_type(tensor=tensor, device=device, *args, **kwargs)
        self.ptype = ptype

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, ptype=Ptype.arithmetic):
        result = MPCTensor([])
        from_shares = ptype.to_tensor().from_shares
        result._tensor = from_shares(share, precision=precision)
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

    def __hash__(self):
        return hash(self.share)

    @property
    def share(self):
        """Returns underlying share"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets share to value"""
        self._tensor.share = value

    @property
    def data(self):
        """Returns share data"""
        return self.share.data

    @data.setter
    def data(self, value):
        """Sets data to value"""
        self.share.data = value

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
        rand._tensor = BinarySharedTensor.rand(
            *sizes, bits=encoder._precision_bits, device=device
        )
        rand._tensor.encoder = encoder
        rand.ptype = Ptype.binary
        return rand.to(Ptype.arithmetic, bits=encoder._precision_bits)

    # Comparators
    @mode(Ptype.binary)
    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1

        precision = 0 if self.encoder.scale == 1 else None
        result = (self >> shift).to(Ptype.arithmetic, precision=precision, bits=1)
        result.encoder._scale = 1
        return result

    @mode(Ptype.arithmetic)
    def eq(self, y):
        """Returns self == y"""
        if comm.get().get_world_size() == 2:
            return (self - y)._eqz_2PC()

        return 1 - self.ne(y)

    @mode(Ptype.arithmetic)
    def ne(self, y):
        """Returns self != y"""
        if comm.get().get_world_size() == 2:
            return 1 - self.eq(y)

        difference = self - y
        difference.share = torch_stack([difference.share, -(difference.share)])
        return difference._ltz().sum(0)

    @mode(Ptype.arithmetic)
    def _eqz_2PC(self):
        """Returns self == 0"""
        # Create BinarySharedTensors from shares
        x0 = MPCTensor(self.share, src=0, ptype=Ptype.binary)
        x1 = MPCTensor(-self.share, src=1, ptype=Ptype.binary)

        # Perform equality testing using binary shares
        x0._tensor = x0._tensor.eq(x1._tensor)
        x0.encoder = self.encoder

        # Convert to Arithmetic sharing
        result = x0.to(Ptype.arithmetic, bits=1)
        result.encoder._scale = 1

        return result

    @mode(Ptype.arithmetic)
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

        if isinstance(y, MPCTensor):
            return result.mul(y.reciprocal())
        result._tensor.div_(y)
        return result

    def div_(self, y):
        """In-place version of :meth:`div`"""
        if isinstance(y, MPCTensor):
            return self.mul_(y.reciprocal())
        self._tensor.div_(y)
        return self

    def index_add(self, dim, index, tensor):
        """Performs out-of-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index.
        """
        result = self.clone()
        assert index.dim() == 1, "index needs to be a vector"
        public = isinstance(tensor, (int, float)) or is_tensor(tensor)
        private = isinstance(tensor, MPCTensor)
        if public:
            result._tensor.index_add_(dim, index, tensor)
        elif private:
            result._tensor.index_add_(dim, index, tensor._tensor)
        else:
            raise TypeError("index_add second tensor of unsupported type")
        return result

    def scatter_add(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor.
        """
        result = self.clone()
        public = isinstance(other, (int, float)) or is_tensor(other)
        private = isinstance(other, CrypTensor)
        if public:
            result._tensor.scatter_add_(dim, index, other)
        elif private:
            result._tensor.scatter_add_(dim, index, other._tensor)
        else:
            raise TypeError("scatter_add second tensor of unsupported type")
        return result

    def scatter(self, dim, index, src):
        """Out-of-place version of :meth:`MPCTensor.scatter_`"""
        result = self.clone()
        if is_tensor(src):
            src = MPCTensor(src)
        assert isinstance(src, MPCTensor), "Unrecognized scatter src type: %s" % type(
            src
        )
        result.share.scatter_(dim, index, src.share)
        return result

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
    "prod": Ptype.arithmetic,
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
