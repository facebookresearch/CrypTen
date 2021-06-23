#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch

from .debug import validation_mode, validate_correctness
from .gradients import (
    AutogradContext,
    BaseAutogradContext,
    get_grad_fn,
    get_grad_fn_registry,
)


# list of all static functions that CrypTensors support:
STATIC_FUNCTIONS = ["cat", "stack"]
STATIC_FUNCTION_MAPPING = {getattr(torch, name): name for name in STATIC_FUNCTIONS}


def _find_all_cryptensors(inputs):
    """
    Recursively find all CrypTensors in an input list, tuple, set, or dict.
    """
    cryptensors = []
    for input in inputs:
        if isinstance(input, CrypTensor):
            cryptensors.append(input)
        elif isinstance(input, (list, tuple, set)):
            cryptensors.extend(_find_all_cryptensors(input))
        elif isinstance(input, dict):
            for value in input.values():
                cryptensors.extend(_find_all_cryptensors(value))
    return cryptensors


class CrypTensorMetaclass(type):
    """
    Metaclass for CrypTensor that ensures autograd is invoked for calls to
    static methods such as `crypten.cat` and `crypten.stack`.
    """

    def __getattribute__(cls, name):
        if name in STATIC_FUNCTIONS:
            dummy = cls([])  # this creates an empty CrypTensor
            dummy.__IS_DUMMY__ = True
            return cls.__getattribute__(dummy, name)
        return type.__getattribute__(cls, name)


class CrypTensor(object, metaclass=CrypTensorMetaclass):
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

    AUTOGRAD_ENABLED = True

    @staticmethod
    @contextmanager
    def no_grad():
        """
        Context manager that disables Crypten's autograd.
        """
        prior_value = CrypTensor.AUTOGRAD_ENABLED
        CrypTensor.set_grad_enabled(False)
        try:
            yield
        finally:
            CrypTensor.set_grad_enabled(prior_value)

    @staticmethod
    @contextmanager
    def enable_grad():
        """
        Context manager that enables Crypten's autograd.
        """
        prior_value = CrypTensor.AUTOGRAD_ENABLED
        CrypTensor.set_grad_enabled(True)
        try:
            yield
        finally:
            CrypTensor.set_grad_enabled(prior_value)

    @staticmethod
    def set_grad_enabled(mode):
        """
        Enables (`mode = True`) or disables (`mode = False`) Crypten's autograd.
        """
        CrypTensor.AUTOGRAD_ENABLED = mode

    def __init__(self, requires_grad=False):
        """
        Creates a new `CrypTensor` object. The `requires_grad` flag determines
        if computations on the created tensor are logged on the autograd tape.

        NOTE: This constructor cannot be called directly. It is only be called
        via `super()` from classes that implement the `CrypTensor` abstraction.
        """
        self.requires_grad = requires_grad  # whether tensors needs gradient
        self._reset_gradients()

    def __new__(cls, *args, **kwargs):
        if cls is CrypTensor:
            raise TypeError("CrypTensor class cannot be instantiated directly.")
        return object.__new__(cls)

    @staticmethod
    def new(*args, **kwargs):
        """
        Static method that creates a new `CrypTensor` of same type.
        """
        raise NotImplementedError("new is not implemented")

    def _reset_gradients(self):
        """Resets gradient information in tensor."""
        self.grad = None  # gradient itself
        self.grad_fn = None  # functions to call for gradient
        self.grad_expected = 0  # number of gradients expected from parents
        self.grad_received = 0  # number of gradients received from parents
        self.children = []  # children of node in graph
        self.ctx = AutogradContext()  # contexts for AutogradFunctions

    def _identify_required_grads(self):
        """Flag all nodes for which gradient needs to be evaluated."""
        self.grad_expected += 1
        if self.grad_expected == 1:  # only backpropagate once from each node
            for child in self.children:
                child._identify_required_grads()

    def backward(self, grad_input=None, top_node=True):
        """
        Backpropagates gradient through the computation graph. The function
        only maintains the gradients in leaf nodes of the graph.
        """
        if self.requires_grad:
            with CrypTensor.no_grad():  # disable autograd for backward pass

                # in initial backward call, identify all required nodes:
                if top_node:
                    self._identify_required_grads()

                # if undefined, set gradient input to one:
                if grad_input is None:
                    if self.nelement() == 1:
                        grad_input = self.new(torch.ones_like(self.share))
                    else:
                        raise RuntimeError(
                            "grad can be implicitly created only for scalar outputs"
                        )

                # process gradient input:
                self.grad_received += 1
                if self.grad is None:
                    self.grad = grad_input  # store gradient...
                else:
                    self.grad.add_(grad_input)  # ... or accumulate gradient

                # if we are in a leaf or if not all parents have backpropagated:
                if len(self.children) == 0 or self.grad_received < self.grad_expected:
                    return  # ... do not proceed.

                # check that we can actually backpropagate:
                if self.grad_fn is None:
                    raise ValueError("Cannot call backward() before forward().")

                # perform backpropagation:
                grad = self.grad_fn.backward(self.ctx, self.grad)
                differentiable_children = [
                    x for x in self.children if self.ctx.is_differentiable(x)
                ]
                self.ctx.reset()  # free up memory used for context

                # call backward function on children:
                if not isinstance(grad, (list, tuple)):
                    grad = (grad,)
                assert len(differentiable_children) <= len(
                    grad
                ), "number of gradients does not match number of children"
                for idx, child in enumerate(differentiable_children):
                    child.backward(grad_input=grad[idx], top_node=False)

                # clean up gradients except in leaf nodes:
                if len(differentiable_children) > 0:
                    self.grad = None

                # remove node from graph:
                self.children = []
                self.grad_expected = 0
                self.grad_received = 0

    def detach_(self):
        """Detaches tensor from the autograd graph (in-place), making it a leaf."""
        self.requires_grad = False
        return self

    def detach(self):
        """Detaches tensor from the autograd graph, making it a leaf."""
        clone = self.clone()
        clone.requires_grad = False
        return clone

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Allows torch static functions to work on CrypTensors."""
        if kwargs is None:
            kwargs = {}
        if func in STATIC_FUNCTION_MAPPING:
            import crypten

            # dispatch torch.{cat,stack} call on CrypTensor to CrypTen:
            return getattr(crypten, STATIC_FUNCTION_MAPPING[func])(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"CrypTen does not support torch function {func}."
            )

    def _get_forward_function_no_ctx(self, grad_fn):

        # determine if self is a dummy object (the case for staticmethods):
        is_dummy = getattr(self, "__IS_DUMMY__", False)

        def autograd_forward_no_ctx(*args, **kwargs):
            if not is_dummy:
                args = [self] + list(args)

            # Create dummy AutogradContext that stores no data
            ctx = BaseAutogradContext()

            with CrypTensor.no_grad():
                result = grad_fn.forward(ctx, *args, **kwargs)
            return result

        return autograd_forward_no_ctx

    def _get_autograd_forward_function(self, name, grad_fn, in_place):

        # determine if self is a dummy object (the case for staticmethods):
        is_dummy = getattr(self, "__IS_DUMMY__", False)

        def autograd_forward(*args, **kwargs):
            """Forward function that stores data for autograd in result."""
            with CrypTensor.no_grad():
                # only CrypTensors can be children:
                tensor_args = _find_all_cryptensors(args)
                children = tensor_args if is_dummy else [self, *tensor_args]

                # identify whether result requires gradient:
                requires_grad = any(child.requires_grad for child in children)

                if not requires_grad:
                    return self.__getattribute__(name)(*args, **kwargs)

                # in-place functions are not supported when requires_grad:
                if in_place:
                    raise RuntimeError("Cannot use in-place functions with autograd.")

                # prepare inputs and context for forward call:
                ctx = AutogradContext()
                if not is_dummy:
                    args = [self] + list(args)

                # apply correct autograd function:
                result = grad_fn.forward(ctx, *args, **kwargs)

                # output may be tensor or tuple
                if not isinstance(result, tuple):
                    result = (result,)
                    remove_tuple = True
                else:
                    remove_tuple = False

                # maintain references to children and context in result:
                for res in result:
                    res.requires_grad = ctx.is_differentiable(res)
                    if res.requires_grad:
                        res.children = children
                        res.grad_fn = grad_fn
                        res.ctx = ctx

                # return result:
                if remove_tuple:
                    result = result[0]
            return result

        return autograd_forward

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
        if name in CrypTensor.PROTECTED_ATTRIBUTES:
            return object.__getattribute__(self, name)

        # Special case for copy_ inplace.
        if name == "copy_":
            return object.__getattribute__(self, "copy_")

        # replace Python built-in methods with corresponding method name:
        name = CrypTensor.PYTHON_BUILTIN.get(name, name)

        # determine inplace and modify name accordingly
        inplace = name.endswith("_") and not name.endswith("__")
        if inplace:
            if CrypTensor.AUTOGRAD_ENABLED and self.requires_grad:
                raise RuntimeError(f"Autograd is not supported for in-place functions.")

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

        # 2. If requires_grad:
        #     a. Fetch from grad_fn.forward; if none exists
        #     b. raise NotImplementedError telling user to use `detach()`
        if CrypTensor.AUTOGRAD_ENABLED:
            if not hasattr(grad_fn, "forward"):
                raise NotImplementedError(
                    f"Autograd forward not implemented for {name}. Please use detach()."
                )
            return self._get_autograd_forward_function(name, grad_fn, inplace)

        # TODO: Add validation_mode / validate_correctness

        # 3. If no_grad or not requires_grad:
        #     a. Try to fetch function from CrypTensor object
        #         - If this fails and function is REQUIRED, raise error
        #     b. Fetch from grad_fn.forward, ignoring AutogradContext

        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name in CrypTensor.REQUIRED_FUNCTIONS:
                raise e
            assert hasattr(grad_fn, "forward")
            return self._get_forward_function_no_ctx(grad_fn)

    # below are all the functions that subclasses of CrypTensor should implement:
    def abs(self):
        raise NotImplementedError("abs is not implemented")

    def __abs__(self):
        return self.abs()

    def __rpow__(self, scalar):
        raise NotImplementedError("__rpow__ is not implemented")

    def pow_(self):
        raise NotImplementedError("pow_ is not implemented")

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

    def add_(self, tensor):
        """Adds :attr:`tensor` to :attr:`self` (in-place) see :meth:`add`."""
        raise NotImplementedError("add_ is not implemented")

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

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def sub_(self, tensor):
        """Subtracts :attr:`tensor` from :attr:`self` (in-place), see :meth:`sub`"""
        raise NotImplementedError("sub_ is not implemented")

    def sub(self, tensor):
        """Subtracts a :attr:`tensor` from :attr:`self` tensor.
        The shape of :attr:`tensor` must be
        `broadcastable`_ with the shape of :attr:`self`.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
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
        """Element-wise multiply with a :attr:`tensor` in-place, see :meth:`mul`."""
        raise NotImplementedError("mul_ is not implemented")

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

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def div_(self, tensor):
        """Element-wise in-place divide by a :attr:`tensor` (see :meth:`div`)."""
        raise NotImplementedError("div_ is not implemented")

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

    def __div__(self, tensor):
        """Element-wise divide by a tensor."""
        return self.div(tensor)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def neg(self):
        r"""
        Returns a new tensor with the negative of the elements of :attr:`self`.

        .. math::
            \text{out} = -1 \times \text{input}
        """
        raise NotImplementedError("neg is not implemented")

    def neg_(self):
        """Negative value of a tensor (in-place), see :meth:`neg`"""
        raise NotImplementedError("neg_ is not implemented")

    def __neg__(self):
        return self.neg()

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

    def norm(self, p="fro", dim=None, keepdim=False):
        """
        Computes the p-norm of the :attr:`self` (or along a dimension)

        Args:
            p (str, int, or float): specifying type of p-norm
            dim (int): optional dimension along which to compute p-norm
            keepdim (bool): whether the output tensor has `dim` retained or not
        """
        raise NotImplementedError("norm is not implemented")

    def mean(self, dim=None):
        """Compute mean."""
        raise NotImplementedError("mean is not implemented")

    def var(self, dim=None):
        """Compute variance."""
        raise NotImplementedError("var is not implemented")

    def conv1d(self, *args, **kwargs):
        """1D convolution."""
        raise NotImplementedError("conv1d is not implemented")

    def conv2d(self, *args, **kwargs):
        """2D convolution."""
        raise NotImplementedError("conv2d is not implemented")

    def avg_pool2d(self, kernel_size, stride=None, padding=0):
        """Perform an average pooling on each 2D matrix of the given tensor

        Args:
            kernel_size (int or tuple): pooling kernel size.
        """
        raise NotImplementedError("avg_pool2d is not implemented")

    def hardtanh(self, min_val=-1, max_val=1):
        r"""Applies the HardTanh function element-wise

        HardTanh is defined as:

        .. math::
            \text{HardTanh}(x) = \begin{cases}
                1 & \text{ if } x > 1 \\
                -1 & \text{ if } x < -1 \\
                x & \text{ otherwise } \\
            \end{cases}

        The range of the linear region :math:`[-1, 1]` can be adjusted using
        :attr:`min_val` and :attr:`max_val`.

        Args:
            min_val: minimum value of the linear region range. Default: -1
            max_val: maximum value of the linear region range. Default: 1
        """
        raise NotImplementedError("hardtanh is not implemented")

    def dot(self, tensor, weights=None):
        """Perform (weighted) inner product with plain or cipher text."""
        raise NotImplementedError("dot is not implemented")

    def ger(self, tensor):
        """Compute outer product."""
        raise NotImplementedError("ger is not implemented")

    def index_add(self, dim, index, tensor):
        """Accumulate the elements of :attr:`tensor` into
        :attr:`self` by adding to the indices in the order given in :attr:`index`

        Example: if ``dim == 0`` and ``index[i] == j``, then the ``i``-th row
        of tensor is added to the ``j``-th row of :attr:`self`

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of tensor to select from
            tensor (MPCTensor or torch.Tensor): containing values to add
        """
        raise NotImplementedError("index_add is not implemented")

    def index_add_(self, dim, index, tensor):
        """Accumulate the elements of :attr:`tensor` into
        :attr:`self` by adding to the indices in the order given in :attr:`index`

        Example: if ``dim == 0`` and ``index[i] == j``, then the ``i``-th row
        of tensor is added to the ``j``-th row of :attr:`self`

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of tensor to select from
            tensor (MPCTensor or torch.Tensor): containing values to add
        """
        raise NotImplementedError("index_add_ is not implemented")

    def scatter_add(self, dim, index, other):
        """Adds all values from the :attr:`other` into :attr:`self` at the
        indices specified in :attr:`index`. This an out-of-place version of
        :meth:`scatter_add_`. For each value in :attr:`other`, it is added to an
        index in :attr:`self` which is specified by its index in :attr:`other`
        for ``dimension != dim`` and by the corresponding
        value in index for ``dimension = dim``.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add,
                can be either empty or the same size of src.
                When empty, the operation returns identity.
            other (Tensor): the source elements to scatter and add
        """
        raise NotImplementedError("scatter_add is not implemented")

    def scatter_add_(self, dim, index, other):
        """Adds all values from the :attr:`other` into :attr:`self` at the
        indices specified in :attr:`index`.
        For each value in :attr:`other`, it is added to an
        index in :attr:`self` which is specified by its index in :attr:`other`
        for ``dimension != dim`` and by the corresponding
        value in index for ``dimension = dim``.


        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add,
                can be either empty or the same size of src.
                When empty, the operation returns identity.
            other (Tensor): the source elements to scatter and add
        """
        raise NotImplementedError("scatter_add_ is not implemented")

    # static methods:
    @staticmethod
    def cat(tensors, dim=0):
        """
        Concatenates a list of `CrypTensor`s along dimension `dim`.
        """
        raise NotImplementedError("cat is not implemented")

    @staticmethod
    def stack(tensors, dim=0):
        """
        Stacks a list of `CrypTensor`s along dimension `dim`.
        """
        raise NotImplementedError("stack is not implemented")

    # Regular functions:
    def clone(self):
        """
        Returns a copy of the :attr:`self` tensor.
        The copy has the same size and data type as :attr:`self`.

        .. note::
            This function is recorded in the computation graph. Gradients
            propagating to the cloned tensor will propagate to the original tensor.
        """
        raise NotImplementedError("clone is not implemented")

    def index_select(self, dim, index):
        """
        Returns a new tensor which indexes the :attr:`self` tensor along
        dimension :attr:`dim` using the entries in :attr:`index`.

        The returned tensor has the same number of dimensions as :attr:`self`
        The dimension :attr:`dim` has the same size as the length
        of :attr:`index`; other dimensions have the same size as in :attr:`self`.
        """
        raise NotImplementedError("index_select is not implemented")

    def take(self, index, dimension=None):
        """
        Returns a new tensor with the elements of :attr:`input` at the given
        indices. When the dimension is None, :attr:`self` tensor is treated as
        if it were viewed as a 1D tensor, and the result takes the same shape as
        the indices. When the dimension is an integer, the result take entries
        of tensor along a dimension according to the :attr:`index`.
        """
        raise NotImplementedError("take is not implemented")

    def pad(self, pad, mode="constant", value=0):
        """Pads tensor with constant."""
        raise NotImplementedError("pad is not implemented")

    # properties:
    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")

    def numel(self):
        raise NotImplementedError("numel is not implemented")

    def nelement(self):
        raise NotImplementedError("nelement is not implemented")

    def dim(self):
        raise NotImplementedError("dim is not implemented")

    def size(self):
        raise NotImplementedError("size is not implemented")

    @property
    def shape(self):
        return self.size()

    def set(self, enc_tensor):
        """Sets self encrypted to enc_tensor in place"""
        raise NotImplementedError("set is not implemented")

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")


from .common import functions

# Register common functions
for module_name in functions.__all__:
    module = getattr(functions, module_name)
    for func in module.__all__:
        setattr(CrypTensor, func, getattr(module, func))


# Register base implementations from gradient functions
for name, grad_fn in get_grad_fn_registry().items():
    if hasattr(grad_fn, "base_impl"):
        setattr(CrypTensor, name, grad_fn.base_impl)
