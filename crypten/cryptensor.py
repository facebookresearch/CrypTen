#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch

from .debug import register_validation
from .gradients import (
    AutogradContext,
    BaseAutogradContext,
    get_grad_fn,
)


class InnerCrypTensor(object):
    """
    Abstract implementation of encrypted tensor type. Every subclass of `CrypTensor`
    must implement the methods defined here. The actual tensor data should live in
    an instance attribute called `_tensor`. When implemented, the `CrypTensor`
    provides a full autograd implementation to the user.
    """

    # attributes that should be dispatched to underlying tensor:
    PROTECTED_ATTRIBUTES = [
        "__dict__",
        "__class__",
        "requires_grad",
        "grad",
        "grad_fn",
        "grad_expected",
        "grad_received",
        "children",
        "ctx",
        "backward",
        "detach",
        "detach_",
        "_reset_gradients",
    ]

    # functions that should be implemented by CrypTensor subclass:
    REQUIRED_FUNCTIONS = [
        "_ltz",
        "add",
        "avg_pool1d",
        "avg_pool2d",
        "clone",
        "conv1d",
        "conv2d",
        "copy_",
        "div_",
        "matmul",
        "neg",
    ]

    # dict for storing functional overrides from subclasses:
    FUNCTION_OVERRIDES = {}

    # mapping of Python built-in methods to CrypTensor methods:
    PYTHON_BUILTIN = {
        "__abs__": "abs",
        "__neg__": "neg",
        "__pow__": "pow",
        "__add__": "add",
        "__radd__": "add",
        "__sub__": "sub",
        "__rsub__": "__rsub__",
        "__mul__": "mul",
        "__rmul__": "mul",
        "__div__": "div",
        "__truediv__": "div",
        "__rtruediv__": "__rtruediv__",
        "__matmul__": "matmul",
        "__imatmul__": "matmul",  # not in-place, matching PyTorch
    }

    # TODO: Automatically register all these functions in CrypTensor?

    def __init__(self, requires_grad=False):
        """
        Creates a new `CrypTensor` object. The `requires_grad` flag determines
        if computations on the created tensor are logged on the autograd tape.

        NOTE: This constructor cannot be called directly. It is only be called
        via `super()` from classes that implement the `CrypTensor` abstraction.
        """
        self.requires_grad = requires_grad

    def __new__(cls, *args, **kwargs):
        if cls is InnerCrypTensor:
            raise TypeError("CrypTensor class cannot be instantiated directly.")
        return object.__new__(cls)

    @register_validation
    def __getattribute__(self, name):
        """
        Makes sure that any function call on the tensor gets recorded in order
        to facilitate gradient computation using autograd.

        For clarity, this function attempts to fetch functions with the following priority:

        1. If name is in PROTECTED_ATTRIBUTES, fetch from the CrypTensor object.

        2. If requires_grad:
            a. Fetch from grad_fn.forward; if none exists
            b. raise NotImplementedError telling user to use `detach()`

        3. If no_grad or not requires_grad:
            a. Try to fetch function from CrypTensor object
                - If this fails and function is REQUIRED, raise error
            b. Fetch from grad_fn.forward, ignoring AutogradContext
        """
        # 1. If name is in PROTECTED_ATTRIBUTES, fetch from the CrypTensor object.
        if name in InnerCrypTensor.PROTECTED_ATTRIBUTES:
            return object.__getattribute__(self, name)

        # Special case for copy_ inplace.
        if name == "copy_":
            return object.__getattribute__(self, "copy_")

        # replace Python built-in methods with corresponding method name:
        name = InnerCrypTensor.PYTHON_BUILTIN.get(name, name)

        # determine inplace and modify name accordingly
        inplace = name.endswith("_") and not name.endswith("__")
        if inplace:
            # Note: native in-place support is now deprecated
            # Instead, CrypTensors now compute out-of-place and
            # copy_ in-place
            name = name[:-1]
            func = self.__getattribute__(name)

            def oop_and_copy(*args, **kwargs):
                result = func(*args, **kwargs)
                self.copy_(result)
                return self

            return oop_and_copy

        # identify the AutogradFunction corresponding to the function name:
        grad_fn = get_grad_fn(name)

        # dispatch calls to size(), etc. without going through AutogradFunction:
        if grad_fn is None:
            return object.__getattribute__(self, name)

        # TODO: Add validation_mode / validate_correctness

        # 3. If no_grad or not requires_grad:
        #     a. Try to fetch function from CrypTensor object
        #         - If this fails and function is REQUIRED, raise error
        #     b. Fetch from grad_fn.forward, ignoring AutogradContext

        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name in InnerCrypTensor.REQUIRED_FUNCTIONS:
                raise e
            assert hasattr(grad_fn, "forward")
            return self._get_forward_function_no_ctx(grad_fn)

    def detach(self):
        """Detaches tensor from the autograd graph, making it a leaf."""
        clone = self.clone()
        clone.requires_grad = False
        return clone

    # Common functions:
    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __abs__(self):
        return self.abs()

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def sub(self, tensor):
        """Subtracts a :attr:`tensor` from :attr:`self` tensor.
        The shape of :attr:`tensor` must be
        `broadcastable`_ with the shape of :attr:`self`.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        return self.add(-tensor)

    def __sub__(self, tensor):
        """Subtracts tensor from this tensor."""
        return self.sub(tensor)

    def __rsub__(self, tensor):
        """Subtracts self from tensor."""
        return -self + tensor

    def __isub__(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        return self.sub_(tensor)

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def __div__(self, tensor):
        """Element-wise divide by a tensor."""
        return self.div(tensor)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def __neg__(self):
        return self.neg()

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def square(self):
        """
        Computes the square of :attr:`self`
        """
        return self * self

    def set(self, enc_tensor):
        """Sets self encrypted to enc_tensor in place"""
        if not isinstance(enc_tensor, InnerCrypTensor):
            enc_tensor = self.new(enc_tensor)
        return self.copy_(enc_tensor)

    @property
    def shape(self):
        return self.size()

    @property
    def device(self):
        return self._tensor.device

    @property
    def data(self):
        return self._tensor.data

    @data.setter
    def data(self, value):
        self._tensor.data = value

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")

    ##############################################################
    # All CrypTensor subclasses should implement the following:  #
    ##############################################################
    def get_plain_text(self):
        """Decrypts the encrypted tensor."""
        raise NotImplementedError("get_plain_text is not implemented")

    def shallow_copy(self):
        """Creates a shallow copy of the CrypTensor."""
        # TODO: Rename this to __copy__()?
        raise NotImplementedError("shallow_copy is not implemented")

    def copy_(self, other):
        """Copies value of other CrypTensor into this CrypTensor."""
        raise NotImplementedError("copy_ is not implemented")

    def clone(self):
        """
        Returns a copy of the :attr:`self` tensor.
        The copy has the same size and data type as :attr:`self`.

        .. note::
            This function is recorded in the computation graph. Gradients
            propagating to the cloned tensor will propagate to the original tensor.
        """
        raise NotImplementedError("clone is not implemented")

    def add(self, tensor):
        r"""Adds :attr:`tensor` to this :attr:`self`.

        Args:
            tensor: can be a torch tensor or a CrypTensor.

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        For a scalar `tensor`,

        .. math::
            \text{{out_i}} = \text{{input_i}} + \text{{tensor}}

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("add is not implemented")

    def mul(self, tensor):
        r"""Element-wise multiply with a :attr:`tensor`.

        .. math::
            \text{out}_i = \text{tensor}_i \times \text{self}_i

        Args:
            tensor (Tensor or float): the tensor or value to multiply.

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("mul is not implemented")

    def div(self, tensor):
        r"""
        Divides each element of :attr:`self` with the :attr:`tensor`
        and returns a new resulting tensor.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{tensor}_i}

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        Args:
            tensor (Tensor or float): the tensor or value in the denominator.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("div is not implemented")

    def neg(self):
        r"""
        Returns a new tensor with the negative of the elements of :attr:`self`.

        .. math::
            \text{out} = -1 \times \text{input}
        """
        raise NotImplementedError("neg is not implemented")

    def matmul(self, tensor):
        r"""Performs matrix multiplication of :attr:`self` with :attr:`tensor`

        The behavior depends on the dimensionality of the tensors as follows:

        - If both tensors are 1-dimensional, the dot product (scalar) is returned.
        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is
          2-dimensional, a 1 is prepended to its dimension for the purpose of
          the matrix multiply. After the matrix multiply, the
          prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is
          1-dimensional, the matrix-vector product is returned.
        - If both arguments are at least 1-dimensional and at least one argument
          is N-dimensional (where N > 2), then a batched matrix multiply is returned.
          If the first argument is 1-dimensional, a 1 is prepended to its dimension
          for the purpose of the batched matrix multiply and removed after.
          If the second argument is 1-dimensional, a 1 is appended to its dimension
          for the purpose of the batched matrix multiple and removed after.
          The non-matrix (i.e. batch) dimensions are broadcasted (and thus
          must be `broadcastable`_).  For example, if :attr:`self` is a
          :math:`(j \times 1 \times n \times m)` tensor and :attr:`tensor` is a
          :math:`(k \times m \times p)` tensor, :attr:`out` will be an
          :math:`(j \times k \times n \times p)` tensor.

        Arguments:
            tensor (Tensor): the tensor to be multiplied

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("matmul is not implemented")

    def conv1d(self, kernel, *args, **kwargs):
        """1D convolution."""
        raise NotImplementedError("conv1d is not implemented")

    def conv2d(self, kernel, *args, **kwargs):
        """2D convolution."""
        raise NotImplementedError("conv2d is not implemented")

    def conv_transpose1d(self, kernel, **kwargs):
        """Perform a 1D transpose convolution (deconvolution) using the given kernel"""
        raise NotImplementedError("conv_transpose1d is not implemented")

    def conv_transpose2d(self, kernel, **kwargs):
        """Perform a 2D transpose convolution (deconvolution) using the given kernel"""
        raise NotImplementedError("conv_transpose2d is not implemented")

    def avg_pool2d(self, kernel_size, stride=None, padding=0):
        """Perform an average pooling on each 2D matrix of the given tensor

        Args:
            kernel_size (int or tuple): pooling kernel size.
        """
        raise NotImplementedError("avg_pool2d is not implemented")

    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        raise NotImplementedError("_ltz is not implemented")

    @staticmethod
    def rand(*sizes, device=None):
        """
        Returns a tensor with elements uniformly sampled in [0, 1). The uniform
        random samples are generated by generating random bits using fixed-point
        encoding and converting the result to an ArithmeticSharedTensor.
        """
        raise NotImplementedError("rand is not implemented")


from torch.utils._pytree import tree_map


def unwrap(e):
    return e.elem if isinstance(e, CrypTensor) else e

def wrap(e):
    return CrypTensor(e) if isinstance(e, InnerCrypTensor) else e

def tree_unwrap(e):
    return tree_map(unwrap, e)

def tree_wrap(e):
    return tree_map(wrap, e)


class AutogradReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        mask = input.gt(0.0)
        ctx.save_for_backward(mask)
        return input.mul(mask)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output.mul(mask)



class CrypTensor(torch.Tensor):
    elem: InnerCrypTensor

    __slots__ = ['elem']

    def __new__(cls, tensor: InnerCrypTensor):
        # TODO: dtype
        self = cls._make_subclass(
            cls, torch.empty(tensor.size(), device='meta'), tensor.requires_grad)
        self.elem = tensor
        return self

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.relu:
            return AutogradReLU.apply(*args, **kwargs)
        with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)

    def __repr__(self):
        return repr(self.elem)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        assert not kwargs

        unwrapped_args = tree_unwrap(args)

        A = torch.ops.aten

        if len(unwrapped_args) >= 1 and isinstance(unwrapped_args[0], InnerCrypTensor):
            self = unwrapped_args[0]
            if func is A.ones_like:
                return wrap(self.new(torch.ones_like(self.data)))
            elif func is A.isnan:
                # TODO: this is wrong (anomaly mode)
                return torch.zeros_like(self.data, dtype=torch.bool)
            # TODO: handle static functions
            else:
                cand_name = func.__name__
                if func is A.rsub:
                    cand_name = '__rsub__'
                elif func is A.mm:
                    cand_name = 'matmul'
                elif func is A._reshape_alias:
                    cand_name = 'reshape'  # todo: technically wrong
                if hasattr(self, cand_name):
                    return tree_wrap(getattr(self, cand_name)(*unwrapped_args[1:]))

        # raise NotImplementedError(f"{func.__name__}({', '.join(map(repr, args))})")
        raise NotImplementedError(f"{func.__name__}({', '.join(map(lambda a: repr(type(a)), args))})")

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            return tree_wrap(getattr(self.elem, name)(*tree_unwrap(args), **tree_unwrap(kwargs)))
        return wrapper

    __CRYPTENSOR_TYPES__ = {}
    __DEFAULT_CRYPTENSOR_TYPE__ = "mpc"

    def new(self, *args, **kwargs):
        return CrypTensor(self.elem.new(*args, **kwargs))

    @staticmethod
    def register_cryptensor(name):
        """Registers a custom :class:`CrypTensor` subclass.

        This decorator allows the user to instantiate a subclass of `CrypTensor`
        from Python cpde, even if the class itself is not  part of CrypTen. To use
        it, apply this decorator to a `CrypTensor` subclass, like this:

        .. code-block:: python

            @CrypTensor.register_cryptensor('my_cryptensor')
            class MyCrypTensor(CrypTensor):
                ...
        """

        def register_cryptensor_cls(cls):
            if name in CrypTensor.__CRYPTENSOR_TYPES__:
                raise ValueError(
                    "Cannot register duplicate CrypTensor type: \
                    tensor type {} already exists.".format(
                        name
                    )
                )
            if not issubclass(cls, InnerCrypTensor):
                raise ValueError(
                    "Registered tensor ({}: {}) must extend \
                    InnerCrypTensor".format(
                        name, cls.__name__
                    )
                )
            CrypTensor.__CRYPTENSOR_TYPES__[name] = cls
            return cls

        return register_cryptensor_cls

    @staticmethod
    @contextmanager
    def no_grad():
        """
        Context manager that disables Crypten's autograd.
        """
        with torch.no_grad():
            yield

    @staticmethod
    @contextmanager
    def enable_grad():
        """
        Context manager that enables Crypten's autograd.
        """
        with torch.enable_grad():
            yield

    @staticmethod
    def set_grad_enabled(mode):
        """
        Enables (`mode = True`) or disables (`mode = False`) Crypten's autograd.
        """
        torch.set_grad_enabled(mode)


from .common import functions

# Register common functions
for module_name in functions.__all__:
    module = getattr(functions, module_name)
    for func in module.__all__:
        setattr(InnerCrypTensor, func, getattr(module, func))
