#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch

from .gradients import AutogradContext, get_grad_fn


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
        if name in ["cat", "stack"]:
            dummy = cls(None)
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

    # attributes that should not be dispatched to underlying tensor:
    PROTECTED_ATTRIBUTES = [
        "__dict__",
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

    def __getattribute__(self, name):
        """
        Makes sure that any function call on the tensor gets recorded in order
        to facilitate gradient computation using autograd.
        """
        if name in CrypTensor.PROTECTED_ATTRIBUTES or not CrypTensor.AUTOGRAD_ENABLED:
            return object.__getattribute__(self, name)
        else:
            # replace Python built-in methods with corresponding method name:
            name = CrypTensor.PYTHON_BUILTIN.get(name, name)

            # identify the AutogradFunction corresponding to the function name:
            grad_fn = get_grad_fn(name)

            # dispatch calls to size(), etc. without going through AutoGradFunction:
            if grad_fn is None:
                return object.__getattribute__(self, name)

            def autograd_forward(*args, **kwargs):
                """Forward function that stores data for autograd in result."""

                # determine if self is a dummy object (the case for staticmethods):
                is_dummy = getattr(self, "__IS_DUMMY__", False)

                # only CrypTensors can be children:
                tensor_args = _find_all_cryptensors(args)
                children = tensor_args if is_dummy else [self, *tensor_args]

                # identify whether result requires gradient:
                requires_grad = any(child.requires_grad for child in children)

                # prepare inputs and context for forward call:
                ctx = AutogradContext()
                if not is_dummy:
                    args = [self] + list(args)

                # apply correct autograd function:
                with CrypTensor.no_grad():
                    result = grad_fn.forward(ctx, *args, **kwargs)
                if not isinstance(result, tuple):  # output may be tensor or tuple
                    result = (result,)
                    remove_tuple = True
                else:
                    remove_tuple = False

                # we only need to build up forward graph if requires_grad is True:
                if requires_grad:

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

    # below are all the functions that subclasses of CrypTensor should implement:
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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
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
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
        raise NotImplementedError("matmul is not implemented")

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def sqrt(self):
        """
        Computes the square root of :attr:`self`
        """
        raise NotImplementedError("sqrt is not implemented")

    def square(self):
        """
        Computes the square of :attr:`self`
        """
        raise NotImplementedError("square is not implemented")

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

    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        raise NotImplementedError("relu is not implemented")

    def argmax(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the maximum value of all elements in
        :attr:`self`

        If multiple values are equal to the maximum, ties will be broken
        (randomly). Note that this deviates from PyTorch's implementation since
        PyTorch does not break ties randomly, but rather returns the lowest
        index of a maximal value.

        If :attr:`keepdim` is True, the output tensor are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors
        having 1 fewer dimension than :attr:`self`.

        If :attr:`one_hot` is True, the output tensor will have the same size as
        the :attr:`self` and contain elements of value `1` on argmax indices
        (with random tiebreaking) and value `0` on other indices.
        """
        raise NotImplementedError("argmax is not implemented")

    def argmin(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the minimum value of all elements in the
        :attr:`self`

        If multiple values are equal to the minimum, ties will be broken
        (randomly). Note that this deviates from PyTorch's implementation since
        PyTorch does not break ties randomly, but rather returns the lowest
        index of a minimal value.

        If :attr:`keepdim` is True, the output tensor are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors
        having 1 fewer dimension than :attr:`self`.

        If :attr:`one_hot` is True, the output tensor will have the same size as
        the :attr:`self` and contain elements of value `1` on argmin indices
        (with random tiebreaking) and value `0` on other indices.
        """
        raise NotImplementedError("argmin is not implemented")

    def max(self, dim=None, keepdim=False, one_hot=False):
        """Returns the maximum value of all elements in :attr:`self`

        If :attr:`dim` is specified, returns a tuple `(values, indices)` where
        `values` is the maximum value of each row of :attr:`self` in the
        given dimension :attr:`dim`. And `indices` is the result of an
        :meth:`argmax` call with the same keyword arguments (:attr:`dim`,
        :attr:`keepdim`, and :attr:`one_hot`)

        If :attr:`keepdim` is True, the output tensors are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors
        having 1 fewer dimension than :attr:`self`.
        """
        raise NotImplementedError("max is not implemented")

    def min(self, dim=None, keepdim=False, one_hot=False):
        """Returns the minimum value of all elements in :attr:`self`.

        If `dim` is sepcified, returns a tuple `(values, indices)` where
        `values` is the minimum value of each row of :attr:`self` tin the
        given dimension :attr:`dim`. And :attr:`indices` is the result of an
        :meth:`argmin` call with the same keyword arguments (:attr:`dim`,
        :attr:`keepdim`, and :attr:`one_hot`)

        If `keepdim` is True, the output tensors are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors
        having 1 fewer dimension than :attr:`self`.
        """
        raise NotImplementedError("min is not implemented")

    def batchnorm(
        self,
        ctx,
        weight,
        bias,
        running_mean=None,
        running_var=None,
        training=False,
        eps=1e-05,
        momentum=0.1,
    ):
        """Batch normalization."""
        raise NotImplementedError("batchnorm is not implemented")

    def conv2d(self, *args, **kwargs):
        """2D convolution."""
        raise NotImplementedError("conv2d is not implemented")

    def max_pool2d(self, kernel_size, padding=None, stride=None, return_indices=False):
        """Applies a 2D max pooling over an input signal composed of several
        input planes.

        If ``return_indices`` is True, this will return the one-hot max indices
        along with the outputs.

        These indices will be returned as with dimensions equal to the
        ``max_pool2d`` output dimensions plus the kernel dimensions. This is
        because each returned index will be a one-hot kernel for each element of
        the output that corresponds to the maximal block element of the
        corresponding input block.

        A max pool with output tensor of size :math:`(i, j, k, l)` with kernel
        size :math:`m` and will return an index tensor of size
        :math:`(i, j, k, l, m, m)`.

        ::

        [ 0,  1,  2,  3]                    [[0, 0], [0, 0]]
        [ 4,  5,  6,  7]         ->         [[0, 1], [0, 1]]
        [ 8,  9, 10, 11]         ->         [[0, 0], [0, 0]]
        [12, 13, 14, 15]                    [[0, 1], [0, 1]]

        Note: This deviates from PyTorch's implementation since PyTorch returns
        the index values for each element rather than a one-hot kernel. This
        deviation is useful for implementing ``_max_pool2d_backward`` later.
        """
        raise NotImplementedError("max_pool2d is not implemented")

    def _max_pool2d_backward(
        self, indices, kernel_size, padding=None, stride=None, output_size=None
    ):
        """Implements the backwards for a `max_pool2d` call where `self` is
        the output gradients and `indices` is the 2nd result of a `max_pool2d`
        call where `return_indices` is True.

        The output of this function back-propagates the gradient (from `self`)
        to be computed with respect to the input parameters of the `max_pool2d`
        call.

        `max_pool2d` can map several input sizes to the same output sizes. Hence,
        the inversion process can get ambiguous. To accommodate this, you can
        provide the needed output size as an additional argument `output_size`.
        Otherwise, this will return a tensor the minimal size that will produce
        the correct mapping.
        """
        raise NotImplementedError("_max_pool2d_backward is not implemented")

    def dropout(self, p=0.5, training=True, inplace=False):
        r"""
        Randomly zeroes some of the elements of the input tensor with
        probability :attr:`p`.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, operation is in-place (default=``False``).
        """
        raise NotImplementedError("dropout is not implemented")

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
            inplace: If set to ``True``, operation is in-place (default=``False``).
        """
        raise NotImplementedError("dropout2d is not implemented")

    def dropout3d(self, p=0.5, training=True, inplace=False):
        r"""
        Randomly zero out entire channels (a channel is a 3D feature map,
        e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
        batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input
        tensor). Each channel will be zeroed out independently on every forward
        call with probability :attr:`p` using samples from a Bernoulli distribution.

        Args:
            p: probability of a channel to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, operation is in-place (default=``False``)
        """
        raise NotImplementedError("dropout3d is not implemented")

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or CrypTensor): values selected at indices
                where condition is False.

        Returns: CrypTensor or torch.tensor
        """
        raise NotImplementedError("where is not implemented")

    def sigmoid(self, reciprocal_method="log"):
        """Computes the sigmoid function on the input value
                sigmoid(x) = (1 + exp(-x))^{-1}
        """
        raise NotImplementedError("sigmoid is not implemented")

    def tanh(self, reciprocal_method="log"):
        """Computes tanh from the sigmoid function:
            tanh(x) = 2 * sigmoid(2 * x) - 1
        """
        raise NotImplementedError("tanh is not implemented")

    def softmax(self, dim, **kwargs):
        """Compute the softmax of a tensor's elements along a given dimension
        """
        raise NotImplementedError("softmax is not implemented")

    def log_softmax(self, dim, **kwargs):
        """Applies a softmax of a tensor's elements along a given dimension,
           followed by a logarithm.
        """
        raise NotImplementedError("log_softmax is not implemented")

    def cos(self):
        """Computes the cosine of the input."""
        raise NotImplementedError("cos is not implemented")

    def sin(self):
        """Computes the sine of the input."""
        raise NotImplementedError("sin is not implemented")

    # Approximations:
    def exp(self):
        """Computes exponential function on the tensor."""
        raise NotImplementedError("exp is not implemented")

    def log(self):
        """Computes the natural logarithm of the tensor."""
        raise NotImplementedError("log is not implemented")

    def reciprocal(self):
        """Computes the reciprocal of the tensor."""
        raise NotImplementedError("reciprocal is not implemented")

    def eq(self, tensor):
        """Element-wise equality

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("eq is not implemented")

    def __eq__(self, tensor):
        """Element-wise equality"""
        return self.eq(tensor)

    def ne(self, tensor):
        """Element-wise inequality

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted boolean tensor containing a True at each location where
            comparison is true.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("ne is not implemented")

    def __ne__(self, tensor):
        """Element-wise inequality"""
        return self.ne(tensor)

    def ge(self, tensor):
        """Element-wise greater than or equal to

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted boolean valued tensor containing a ``True`` at each
            location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("ge is not implemented")

    def __ge__(self, tensor):
        """Element-wise greater than or equal to"""
        return self.ge(tensor)

    def gt(self, tensor):
        """Element-wise greater than

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted boolean valued tensor containing a True at each
            location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("gt is not implemented")

    def __gt__(self, tensor):
        """Element-wise greater than"""
        return self.gt(tensor)

    def le(self, tensor):
        """Element-wise less than or equal to

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted boolean valued tensor containing a True at each
            location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("le is not implemented")

    def __le__(self, tensor):
        """Element-wise less than or equal to"""
        return self.le(tensor)

    def lt(self, tensor):
        """Element-wise less than

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted boolean valued tensor containing a True at each
            location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B509
        raise NotImplementedError("lt is not implemented")

    def __lt__(self, tensor):
        """Element-wise less than"""
        return self.lt(tensor)

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

    def __getitem__(self, index):
        """
        Returns an encrypted tensor containing elements of self at `index`
        """
        raise NotImplementedError("__getitem__ is not implemented")

    def __setitem__(self, index, value):
        """
        Sets elements of `self` at index `index` to `value`.
        """
        raise NotImplementedError("__setitem__ is not implemented")

    def index_select(self, dim, index):
        """
        Returns a new tensor which indexes the :attr:`self` tensor along
        dimension :attr:`dim` using the entries in :attr:`index`.

        The returned tensor has the same number of dimensions as :attr:`self`
        The dimension :attr:`dim` has the same size as the length
        of :attr:`index`; other dimensions have the same size as in :attr:`self`.
        """
        raise NotImplementedError("index_select is not implemented")

    def view(self, *shape):
        r"""
        Returns a new encrypted tensor with the same data as the :attr:`self`
        tensor but of a different shape.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed,
        the new view size must be compatible with its original size and stride,
        i.e., each new view dimension must either be a subspace of an original
        dimension, or only span across original dimensions
        :math:`d, d+1, \dots, d+k` that satisfy the following contiguity-like
        condition that :math:`\forall i = 0, \dots, k-1`,

        .. math::
            \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Args:
            shape (torch.Size or int...): the desired
        """
        raise NotImplementedError("view is not implemented")

    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens a contiguous range of dims in a tensor.

        Args:
            start_dim (int): the first dim to flatten. Default is 0.
            end_dim (int): the last dim to flatten. Default is -1.
        """
        raise NotImplementedError("flatten is not implemented")

    def t(self):
        """
        Expects :attr:`self` to be <= 2D tensor and transposes dimensions 0 and 1.

        0D and 1D tensors are returned as is and for 2D tensors this can be
        seen as a short-hand function for `self.transpose(0, 1)`.
        """
        raise NotImplementedError("t is not implemented")

    def transpose(self, dim0, dim1):
        """
        Returns a tensor that is a transposed version of :attr:`self`
        The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

        The resulting tensor shares it’s underlying storage with :attr:`self`,
        so changing the content of one would change the content of the
        other.

        Args:
            dim0 (int): the first dimension to be transposed
            dim1 (int): the second dimension to be transposed
        """
        raise NotImplementedError("t is not implemented")

    def unsqueeze(self, dim):
        """
        Returns a new tensor with a dimension of size one inserted at the
        specified position.

        The returned tensor shares the same underlying data with :attr:`self`

        A :attr:`dim` value within the range `[-self.dim() - 1, self.dim() + 1)`
        can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
        applied at `dim = dim + self.dim() + 1`.

        Args:
            dim (int): the index at which to insert the singleton dimension
        """
        raise NotImplementedError("unsqueeze is not implemented")

    def squeeze(self, dim=None):
        """
        Returns a tensor with all the dimensions of :attr:`self` of size 1 removed.

        For example, if :attr:`self` is of shape:
        `(A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D)` then the
        returned tensor will be of shape: `(A \times B \times C \times D)(A×B×C×D)`.

        When :attr:`dim` is given, a :meth:`squeeze` operation is done only in
        the given dimension. If :attr:`self` is of shape:
        `(A \times 1 \times B)(A×1×B)`, `squeeze(self, 0)` leaves the tensor
        unchanged, but `squeeze(self, 1)` will squeeze the tensor to the
        shape `(A \times B)(A×B)`.
        """
        raise NotImplementedError("squeeze is not implemented")

    def repeat(self, *sizes):
        """
        Repeats :attr:`self` along the specified dimensions.

        Unlike expand(), this function copies the tensor’s data.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this
            tensor along each dimension
        """
        raise NotImplementedError("repeat is not implemented")

    def narrow(self, dim, start, length):
        """
        Returns a new tensor that is a narrowed version of :attr:`self`
        The dimension :attr:`dim` is input from :attr:`start` to
        :attr:`start + length`.  The returned tensor and :attr:`self` share the
        same underlying storage.
        """
        raise NotImplementedError("narrow is not implemented")

    def expand(self, *sizes):
        """
        Returns a new view of :attr:`self` with singleton dimensions
        expanded to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the size
        cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a new
        view on the existing tensor where a dimension of size one is expanded to
        a larger size by setting the :attr:`stride` to 0. Any dimension of size
        1 can be expanded to an arbitrary value without allocating new memory.
        """
        raise NotImplementedError("expand is not implemented")

    def roll(self, shifts, dims=None):
        """
        Roll :attr:`self` along the given dimensions :attr:`dims`. Elements that
        are shifted beyond the last position are re-introduced at the first
        position. If a dimension is not specified, the tensor will be flattened
        before rolling and then restored to the original shape.
        """
        raise NotImplementedError("roll is not implemented")

    def unfold(self, dimension, size, step):
        """
        Returns a tensor which contains all slices of size :attr:`size` from
        :attr:`self` in the dimension :attr:`dimension`.

        Step between two slices is given by :attr:`step`

        If `sizedim` is the size of :attr:`dimension` for :attr:`self`, the size
        of :attr:`dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size :attr:`size` is appended in the returned
        tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice
        """
        raise NotImplementedError("unfold is not implemented")

    def take(self, index, dimension=None):
        """
        Returns a new tensor with the elements of :attr:`input` at the given
        indices. When the dimension is None, :attr:`self` tensor is treated as
        if it were viewed as a 1D tensor, and the result takes the same shape as
        the indices. When the dimension is an integer, the result take entries
        of tensor along a dimension according to the :attr:`index`.
        """
        raise NotImplementedError("take is not implemented")

    def flip(self, input, dims):
        """
        Reverse the order of a n-D tensor along given axis in dims.

        Args:
            dims (a list or tuple): axis to flip on
        """
        raise NotImplementedError("flip is not implemented")

    def pad(self, pad, mode="constant", value=0):
        """Pads tensor with constant."""
        raise NotImplementedError("pad is not implemented")

    def trace(self):
        """
        Returns the sum of the elements of the diagonal of :attr:`self`.
        :attr:`self` has to be a 2D tensor.
        """
        raise NotImplementedError("trace is not implemented")

    def sum(self, dim=None, keepdim=False):
        """
        Returns the sum of all elements in the :attr:`self`

        If :attr:`dim` is a list of dimensions,
        reduce over all of them.
        """
        raise NotImplementedError("sum is not implemented")

    def cumsum(self, dim):
        """
        Returns the cumulative sum of elements of :attr:`self` in the dimension
        :attr:`dim`

        For example, if :attr:`self` is a vector of size N, the result will also
        be a vector of size N, with elements.

        .. math::
            y_i = x_1 + x_2 + x_3 + \dots + x_i

        Args:
            dim  (int): the dimension to do the operation over
        """  # noqa: W605
        raise NotImplementedError("cumsum is not implemented")

    def reshape(self, shape):
        """
        Returns a tensor with the same data and number of elements as
        :attr:`self` but with the specified :attr:`shape`.

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        raise NotImplementedError("reshape is not implemented")

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by :attr:`dim`.

        For a 3-D tensor the output is specified by:
            - out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            - out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            - out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        """
        raise NotImplementedError("reshape is not implemented")

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
