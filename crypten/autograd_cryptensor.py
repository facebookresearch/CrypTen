#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .gradients import get_grad_fn


class AutogradContext(object):
    """
    Object that can be used by AutogradFunction for saving context information.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.context = []
        self.non_differentiable = []

    def save_for_backward(self, value):
        self.context.append(value)

    def save_multiple_for_backward(self, values):
        for value in values:
            self.save_for_backward(value)

    def mark_non_differentiable(self, non_differentiable):
        if not isinstance(non_differentiable, list):
            non_differentiable = [non_differentiable]
        self.non_differentiable.extend(id(x) for x in non_differentiable)

    def is_differentiable(self, tensor):
        return id(tensor) not in self.non_differentiable

    @property
    def saved_tensors(self):
        return self.context


class AutogradCrypTensor(object):
    """
    CrypTensor with support for autograd, akin to the `Variable` originally in
    PyTorch.
    """

    # attributes that should not be dispatched to underlying tensor:
    PROTECTED_ATTRIBUTES = [
        "__dict__",
        "_tensor",
        "requires_grad",
        "grad",
        "grad_fn",
        "grad_computed",
        "parents",
        "children",
        "ctx",
        "backward",
        "detach",
        "detach_",
        "_reset_gradients",
        "tensor",
    ]

    def __init__(self, tensor, requires_grad=True):
        if torch.is_tensor(tensor):
            raise ValueError("Cannot create AutogradCrypTensor from PyTorch tensor.")
        self._tensor = tensor  # value of tensor
        self.requires_grad = requires_grad  # whether tensors needs gradient
        self._reset_gradients()

    def _reset_gradients(self):
        """Resets gradient information in tensor."""
        self.grad = None  # gradient itself
        self.grad_fn = None  # functions to call for gradient
        self.grad_computed = False  # whether gradient already computed
        self.parents = []  # parents of node in graph
        self.children = []  # children of node in graph
        self.ctx = AutogradContext()  # contexts for AutogradFunctions

    @property
    def tensor(self):
        """Returns underlying (non-autograd) tensor."""
        return self._tensor

    def backward(self, grad_input=None, top_node=True):
        """
        Backpropagates gradient through the computation graph. The function
        only maintains the gradients in leaf nodes of the graph.
        """
        if self.requires_grad:

            # if we are in a leaf or if not all parents have backpropagated:
            parents_done = all(parent.grad_computed for parent in self.parents)
            if len(self.children) == 0 or (not top_node and not parents_done):
                # Set grad_input to correct dimension and size if not already
                grad_input = grad_input.view(self.size())
                if self.grad is None:
                    self.grad = grad_input  # store gradient...
                else:
                    self.grad.add_(grad_input)  # ... or accumulate gradient...
                return  # ... and do not proceed.

            # if undefined, set gradient input to all ones:
            if grad_input is None:
                grad_input = self._tensor.new(torch.ones(self._tensor.size()))

            # check that we can actually backpropagate:
            if self.grad_fn is None:
                raise ValueError("Cannot call backward() before forward().")
            if self.grad_fn is None:
                raise NotImplementedError(
                    "Gradient for {} not implemented.".format(self.grad_fn)
                )

            # perform backpropagation:
            grad = self.grad_fn.backward(self.ctx, grad_input)
            self.grad_computed = True  # mark gradient as computed
            differentiable_children = [
                x for x in self.children if self.ctx.is_differentiable(x._tensor)
            ]

            self.ctx.reset()  # free up memory used for context
            if not isinstance(grad, (list, tuple)):
                grad = (grad,)

            assert len(differentiable_children) <= len(
                grad
            ), "number of gradients to backpropagate does not match number of children"
            for idx, child in enumerate(differentiable_children):
                child.backward(grad_input=grad[idx], top_node=False)

            # clean up gradients except in leaf nodes:
            if len(differentiable_children) > 0:
                self.grad = None

            # remove node from graph:
            self.parents = []
            self.children = []

    def detach_(self):
        """Detaches tensor from the autograd graph (in-place), making it a leaf."""
        self.requires_grad = False
        return self

    def detach(self):
        """Detaches tensor from the autograd graph, making it a leaf."""
        return AutogradCrypTensor(self._tensor.clone(), requires_grad=False)

    def __getattribute__(self, name):
        """
        Makes sure that any function call on the tensor gets recorded in order
        to facilitate gradient computation using autograd.
        """
        if name in AutogradCrypTensor.PROTECTED_ATTRIBUTES:
            return object.__getattribute__(self, name)
        else:
            # replace pytorch buildins with corresponding functions
            name = PYTORCH_BUILTIN.get(name, name)

            # determine if we are applying an autograd function:
            grad_fn = get_grad_fn(name)

            # dispatch calls to size(), etc. to underlying CrypTensor:
            if grad_fn is None:
                return getattr(self.__dict__["_tensor"], name)

            def autograd_forward(*args, **kwargs):
                """Forward function that stores data for autograd in result."""

                # mark gradient as not computed:
                self.grad_computed = False

                # only AutogradCrypTensors can be children:
                tensor_args, non_autograd_found = [], False
                for arg in args:
                    if isinstance(arg, AutogradCrypTensor):
                        if non_autograd_found:
                            raise ValueError(
                                "In the inputs, an object that is not an "
                                "AutogradCrypTensor cannot be followed by an "
                                "AutogradCrypTensor."
                            )  # backward() assumes this when iterating over children
                        tensor_args.append(arg)
                    else:
                        non_autograd_found = True

                # identify children and whether result requires gradient:
                children = [self, *tensor_args]
                requires_grad = any(child.requires_grad for child in children)

                # prepare inputs and context for forward call:
                ctx = AutogradContext()
                inputs = [self] + list(args)
                inputs = [
                    input._tensor if isinstance(input, AutogradCrypTensor) else input
                    for input in inputs
                ]
                if len(inputs) == 1:
                    inputs = inputs[0]  # unpack input list if possible

                # apply correct autograd function:
                result = grad_fn.forward(ctx, inputs, **kwargs)
                if not isinstance(result, tuple):  # output may be tensor or tuple
                    result = (result,)
                    remove_tuple = True
                else:
                    remove_tuple = False

                # wrap results and maintain references to children and context:
                result = tuple(
                    AutogradCrypTensor(res, requires_grad=False) for res in result
                )
                for res in result:
                    res.requires_grad = requires_grad and ctx.is_differentiable(
                        res.tensor
                    )
                    if res.requires_grad:
                        res.children = children
                        res.grad_fn = grad_fn
                        res.ctx = ctx
                    self.parents.append(res)

                # return result:
                if remove_tuple:
                    result = result[0]
                return result

            return autograd_forward


# register all Python built-in functions in AutogradCrypTensor:
def register_python_builtin(name, value):
    def fn(self, *args, **kwargs):
        return getattr(self, value)(*args, **kwargs)

    setattr(AutogradCrypTensor, name, fn)


PYTORCH_BUILTIN = {
    "__abs__": "abs",
    "__neg__": "neg",
    "__pow__": "pow",
    "__rpow__": "pow",
    "__add__": "add",
    "__radd__": "add",
    "__iadd__": "add_",
    "__sub__": "sub",
    "__rsub__": "__rsub__",
    "__isub__": "sub_",
    "__mul__": "mul",
    "__rmul__": "mul",
    "__imul__": "mul_",
    "__div__": "div",
    "__truediv__": "div",
    "__rtruediv__": "__rtruediv__",
    "__itruediv__": "div_",
    "__matmul__": "matmul",
    "__imatmul__": "matmul",  # not in-place, matching PyTorch
    "__eq__": "eq",
    "__ne__": "ne",
    "__ge__": "ge",
    "__gt__": "gt",
    "__le__": "le",
    "__lt__": "lt",
}  # TODO: Add unit tests for these built-in functions.
for name, value in PYTORCH_BUILTIN.items():
    register_python_builtin(name, value)
