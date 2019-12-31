#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce

import crypten
import torch


# registry that maps function names to AutogradFunctions:
FUNCTION_REGISTRY = {}


def register_function(name):
    """Decorator that registers a new autograd function."""

    def register_function_cls(cls):
        """Function performing the actual registration."""
        if name in FUNCTION_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        if not issubclass(cls, AutogradFunction):
            raise ValueError(
                "Function (%s: %s) must extend AutogradFunction" % (name, cls.__name__)
            )
        cls.name = name
        FUNCTION_REGISTRY[name] = cls
        return cls

    return register_function_cls


def get_grad_fn(name):
    """
    Returns gradient function for the CrypTen function with the specified name.
    """
    if name.endswith("_") and not name.endswith("__"):  # TODO: Make less hacky.
        raise NotImplementedError("Autograd on in-place operations not supported.")
    if name in FUNCTION_REGISTRY:
        return FUNCTION_REGISTRY[name]
    else:
        return None


def _ensure_tensors(inputs):
    """
    Converts scalars in inputs to correct tensor type.
    """
    for idx in range(len(inputs)):
        if isinstance(inputs[idx], (int, float)):
            inputs[idx] = torch.tensor(inputs[idx])
    return inputs


def _inverse_broadcast(grad_output, input_size):
    """
    Performs the inverse operation of a broadcast.
    """

    # special case where input was a scalar:
    if input_size == torch.Size():
        return grad_output.sum()

    # remove leading dimensions:
    while grad_output.dim() > len(input_size):
        grad_output = grad_output.sum(0, keepdim=False)
    assert grad_output.dim() == len(input_size), "cannot perform inverse broadcast"

    # perform accumulation across broadcast dimensions:
    for dim in range(grad_output.dim()):
        if input_size[dim] == 1 and grad_output.size(dim) > 1:
            grad_output = grad_output.sum(dim, keepdim=True)
    return grad_output


class AutogradFunction(object):
    """
    Base implementation of a function that supports autograd.
    """

    @staticmethod
    def forward(ctx, input):
        raise NotImplementedError("Forward function not implemented.")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward function not implemented.")

    def __str__(self):
        if hasattr(self, "name"):
            return self.name


@register_function("t")
class AutogradT(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.t()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.t()


@register_function("transpose")
class AutogradTranspose(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim1, dim2 = input
        ctx.save_multiple_for_backward((dim1, dim2))
        return input.transpose(dim1, dim2)

    @staticmethod
    def backward(ctx, grad_output):
        dim1, dim2 = ctx.saved_tensors
        return grad_output.transpose(dim2, dim1)


@register_function("flip")
class AutogradFlip(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dims = input
        ctx.save_for_backward(dims)
        return input.flip(dims)

    @staticmethod
    def backward(ctx, grad_output):
        dims, = ctx.saved_tensors
        return grad_output.flip(dims)


@register_function("clone")
class AutogradClone(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


@register_function("cat")
class AutogradCat(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        ctx.save_multiple_for_backward((dim, [t.size(dim) for t in input]))
        return crypten.cat(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, split_sections = ctx.saved_tensors
        return grad_output.split(split_sections, dim=dim)


@register_function("stack")
class AutogradStack(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        ctx.save_for_backward(dim)
        return crypten.stack(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, = ctx.saved_tensors
        return grad_output.unbind(dim=dim)


@register_function("view")
class AutogradView(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, *size = input
        ctx.save_for_backward(input)
        return input.view(*size)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.view(input.size())


@register_function("reshape")
class AutogradReshape(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, shape = input
        ctx.save_for_backward(input.size())
        return input.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        size, = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("flatten")
class AutogradFlatten(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.size())
        return input.flatten()

    @staticmethod
    def backward(ctx, grad_output):
        size, = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("narrow")
class AutogradNarrow(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim, start, length = input
        ctx.save_multiple_for_backward((input.size(dim), dim, start, length))
        return input.narrow(dim, start, length)

    @staticmethod
    def backward(ctx, grad_output):
        size, dim, start, length = ctx.saved_tensors

        # pad is applied to dimensions in reverse order
        dim = grad_output.dim() - 1 - dim

        # pad is applied in pairs that denote the pads at the beginning and end
        # of the tensor along the given dimension
        pad = [0] * 2 * grad_output.dim()
        pad[2 * dim] = start
        pad[2 * dim + 1] = size - length - start
        return grad_output.pad(pad)


@register_function("take")
class AutogradTake(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        # support input without dimension to match PyTorch
        if len(input) == 2:
            input, index = input
            dimension = None
        else:
            input, index, dimension = input
        ctx.save_multiple_for_backward((input.size(), index, dimension))
        return input.take(index, dimension)

    @staticmethod
    def backward(ctx, grad_output):
        size, index, dimension = ctx.saved_tensors
        grad = grad_output.new(torch.zeros(size))
        if dimension is None:
            grad_flat = grad.flatten()
            flat_index = index.flatten()
            grad_output_flat = grad_output.flatten()
            grad_flat[flat_index] = grad_output_flat
            grad = grad_flat.reshape(size)
        else:
            flat_index = index.flatten()
            grad_output_flat = grad_output.flatten(
                start_dim=dimension, end_dim=(dimension + index.dim() - 1)
            )
            grad.index_add_(dimension, flat_index, grad_output_flat)
        return grad


@register_function("gather")
class AutogradGather(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim, index = input
        ctx.save_multiple_for_backward([input.size(), dim, index])
        return input.gather(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        size, dim, index = ctx.saved_tensors
        return grad_output.new(torch.zeros(size)).scatter_add_(dim, index, grad_output)


@register_function("scatter")
class AutogradScatter(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim, index, src = input
        output = input.scatter(dim, index, src)
        ctx.save_multiple_for_backward([dim, index])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dim, index = ctx.saved_tensors

        size = grad_output.size()
        mask = torch.ones(size).scatter(dim, index, torch.zeros(size)).long()
        input_grad = grad_output.mul(mask)
        src_grad = grad_output.gather(dim, index)
        return (input_grad, src_grad)


@register_function("roll")
class AutogradRoll(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        if len(input) < 3:
            input.append(None)
        input, shifts, dims = input
        ctx.save_multiple_for_backward((shifts, dims))
        return input.roll(shifts, dims=dims)

    @staticmethod
    def backward(ctx, grad_output):
        shifts, dims = ctx.saved_tensors

        # Reverse and negate shifts
        if isinstance(shifts, (tuple, list)):
            shifts = list(shifts)
            for i, shift in enumerate(shifts):
                shifts[i] = -shift
            shifts.reverse()
        else:
            shifts = -shifts

        # Reverse dims
        if isinstance(dims, (tuple, list)):
            dims = list(dims)
            dims.reverse()

        return grad_output.roll(shifts, dims)


@register_function("squeeze")
class AutogradSqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, input):

        # preprocess inputs:
        dim = None
        if isinstance(input, (tuple, list)) and len(input) == 1:
            input, = input  # no dimension to squeeze specified
        elif isinstance(input, (tuple, list)):
            input, dim = input  # dimension to squeeze specified

        # perform the actual squeeze:
        output = input.squeeze() if dim is None else input.squeeze(dim)

        # keep correct dimensions for backward pass:
        if dim is None:
            dims = [idx for idx, sz in enumerate(output.size()) if sz == 1]
        else:
            dims = [dim]
        ctx.save_for_backward(dims)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dims, = ctx.saved_tensors
        grad_input = grad_output
        for dim in dims:
            grad_input = grad_input.unsqueeze(dim)
        return grad_input


@register_function("unsqueeze")
class AutogradUnsqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim = input
        ctx.save_for_backward(dim)
        return input.unsqueeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, = ctx.saved_tensors
        return grad_output.squeeze(dim)


@register_function("__getitem__")
class AutogradGetItem(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, index = input
        ctx.save_multiple_for_backward([input.size(), index])
        return input[index]

    @staticmethod
    def backward(ctx, grad_output):
        size, index = ctx.saved_tensors
        grad = grad_output.new(torch.zeros(size))
        grad[index] = grad_output
        return grad


@register_function("neg")
class AutogradNeg(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.neg()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


@register_function("relu")
class AutogradReLU(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        mask = input.gt(0.0)
        ctx.save_for_backward(mask)
        return input.mul(mask)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output.mul(mask)


@register_function("dropout")
class AutogradDropout(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, inplace=False):
        rand_tensor = crypten.mpc.rand(input.size())
        boolean_mask = rand_tensor > p
        if inplace:
            result = input.mul_(boolean_mask).div_(1 - p)
        else:
            result = input.mul(boolean_mask).div_(1 - p)
        ctx.save_multiple_for_backward([boolean_mask, p])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        boolean_mask, p = ctx.saved_tensors
        return grad_output.mul(boolean_mask.div(1 - p))


@register_function("_feature_dropout")
class AutogradFeatureDropout(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, inplace=False):
        feature_dropout_size = input.size()[0:2]
        rand_tensor = crypten.mpc.rand(feature_dropout_size)
        boolean_mask = rand_tensor > p
        for i in range(2, input.dim()):
            boolean_mask = boolean_mask.unsqueeze(i)
        boolean_mask.share, tensor = torch.broadcast_tensors(
            boolean_mask.share, input.share
        )
        if inplace:
            result = input.mul_(boolean_mask).div_(1 - p)
        else:
            result = input.mul(boolean_mask).div_(1 - p)
        ctx.save_multiple_for_backward([boolean_mask, p])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        boolean_mask, p = ctx.saved_tensors
        return grad_output.mul(boolean_mask.div(1 - p))


@register_function("dropout2d")
class AutogradDropout2d(AutogradFeatureDropout):
    pass


@register_function("dropout3d")
class AutogradDropout3d(AutogradFeatureDropout):
    pass


@register_function("tanh")
class AutogradTanh(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        activations = input.tanh()
        ctx.save_for_backward(activations)
        return activations

    @staticmethod
    def backward(ctx, grad_output):
        activations, = ctx.saved_tensors
        return grad_output.mul(activations.square().neg().add(1.0))


@register_function("add")
class AutogradAdd(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input = _ensure_tensors(input)
        ctx.save_multiple_for_backward([input[0].size(), input[1].size()])
        return input[0].add(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("sub")
class AutogradSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input = _ensure_tensors(input)
        ctx.save_multiple_for_backward([input[0].size(), input[1].size()])
        return input[0].sub(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2).neg(),
        )


@register_function("__rsub__")
class AutogradRSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input = _ensure_tensors(input)
        ctx.save_multiple_for_backward([input[0].size(), input[1].size()])
        return (-input[0]).add(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1).neg(),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("mul")
class AutogradMul(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input = _ensure_tensors(input)
        ctx.save_for_backward(input)
        return input[0].mul(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        self_, other = input
        return (
            _inverse_broadcast(grad_output.mul(other), self_.size()),
            _inverse_broadcast(grad_output.mul(self_), other.size()),
        )


@register_function("matmul")
class AutogradMatMul(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[0].matmul(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        self_, other = input
        return (
            _inverse_broadcast(
                grad_output.matmul(other.transpose(-2, -1)), self_.size()
            ),
            _inverse_broadcast(
                self_.transpose(-2, -1).matmul(grad_output), other.size()
            ),
        )


@register_function("div")
class AutogradDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, other = input
        if crypten.is_encrypted_tensor(other):
            other_reciprocal = other.reciprocal()
            ctx.save_multiple_for_backward([input, other_reciprocal])
            return input.mul(other_reciprocal)
        else:
            ctx.save_multiple_for_backward([input.size(), other])
            return input.div(other)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors

        # saved is a list of [input, other_reciprocal]
        if crypten.is_encrypted_tensor(saved[1]):
            input, other_reciprocal = saved
            grad_input = other_reciprocal.mul(grad_output)
            grad_other = other_reciprocal.square().mul(input).mul(grad_output).neg()
            return (
                _inverse_broadcast(grad_input, input.size()),
                _inverse_broadcast(grad_other, other_reciprocal.size()),
            )
        # saved is a public tensor or scalar
        else:
            input_size, other = saved
            grad_input = grad_output.div(other)
            if torch.is_tensor(other):
                return _inverse_broadcast(grad_input, input_size)
            else:
                return grad_input


@register_function("__rtruediv__")
class AutogradRDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, other = input
        reciprocal = input.reciprocal()
        ctx.save_multiple_for_backward([reciprocal, other])
        return reciprocal.mul(other)

    @staticmethod
    def backward(ctx, grad_output):
        reciprocal, other = ctx.saved_tensors
        grad_input = reciprocal.square().mul(other).mul(grad_output).neg()
        grad_input = _inverse_broadcast(grad_input, reciprocal.size())

        if torch.is_tensor(other) or crypten.is_encrypted_tensor(other):
            grad_other = reciprocal.mul(grad_output)
            grad_other = _inverse_broadcast(grad_other, other.size())
            return (grad_input, grad_other)
        else:
            return grad_input


@register_function("pow")
class AutogradPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input, power = input
        return input.pow(power)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input, power = input
        return input.pow(power - 1.0).mul_(power).mul_(grad_output)


@register_function("pos_pow")
class AutogradPosPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, power = input
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            ctx.save_multiple_for_backward([input, power])
            return input.pow(power)
        else:
            log_input = input.log()
            ctx.save_multiple_for_backward([log_input, power])
            return log_input.mul(power).exp()

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.saved_tensors
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            return input.pow(power - 1.0).mul_(power).mul_(grad_output)
        else:
            return input.mul(power - 1.0).mul_(power).exp().mul(grad_output)


@register_function("square")
class AutogradSquare(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.square()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.mul(input.mul(2.0))


@register_function("sqrt")
class AutogradSqrt(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        output = input.sqrt()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output.div(output.mul_(2.0))


@register_function("exp")
class AutogradExp(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        output = input.exp()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return output.mul(grad_output)


@register_function("log")
class AutogradLog(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.log()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.div(input)


@register_function("reciprocal")
class AutogradReciprocal(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        reciprocal = input.reciprocal()
        ctx.save_for_backward(reciprocal)
        return reciprocal

    @staticmethod
    def backward(ctx, grad_output):
        reciprocal, = ctx.saved_tensors
        return grad_output.neg().mul_(reciprocal).mul_(reciprocal)


@register_function("dot")
class AutogradDot(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[0].dot(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        self_, other = input
        return (grad_output.mul(other), grad_output.mul(self_))


@register_function("ger")
class AutogradGer(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input[0].ger(input[1])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input, other = input
        return (grad_output.matmul(other), input.matmul(grad_output))


@register_function("sin")
class AutogradSin(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        cossin = input.cossin()
        ctx.save_for_backward(cossin[0])
        return cossin[1]

    @staticmethod
    def backward(ctx, grad_output):
        cos, = ctx.saved_tensors
        return grad_output.mul(cos)


@register_function("cos")
class AutogradCos(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        cossin = input.cossin()
        ctx.save_for_backward(cossin[1])
        return cossin[0]

    @staticmethod
    def backward(ctx, grad_output):
        sin, = ctx.saved_tensors
        return grad_output.mul(sin.neg_())


@register_function("abs")
class AutogradAbs(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        sign = input.sign()
        ctx.save_for_backward(sign)
        return input.mul(sign)

    @staticmethod
    def backward(ctx, grad_output):
        sign, = ctx.saved_tensors
        return grad_output.mul(sign.mul_(2.0).sub_(1.0))


@register_function("sign")
class AutogradSign(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.sub(grad_output)


@register_function("norm")
class AutogradNorm(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p="fro", dim=None, keepdim=False):
        if p == float("inf"):
            sign = input.sign()
            if dim is None:
                input = input.mul(sign)
                argmax = input.argmax(one_hot=True)
                max = input.mul(argmax).sum()
            else:
                max, argmax = input.mul(sign).max(dim, keepdim=keepdim, one_hot=True)

            ctx.save_multiple_for_backward((sign, argmax, p, dim, keepdim))
            return max
        else:
            if dim is None:
                norm = input.norm(p=p)
            else:
                norm = input.norm(p=p, dim=dim, keepdim=keepdim)
            ctx.save_multiple_for_backward((input, norm, p, dim, keepdim))
            return norm

    @staticmethod
    def backward(ctx, grad_output):
        input, norm, p, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)

        if p == 2 or p == "fro":
            return grad_output.mul(input.div(norm))
        elif p == float("inf"):
            sign, argmax = input, norm
            return grad_output.mul(argmax).mul(sign)
        else:
            sign = input.sign()
            abs = input.mul(sign)
            return grad_output.mul(abs.div(norm).pos_pow(p - 1).mul(sign))


@register_function("sum")
class AutogradSum(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.sum(dim=dim, keepdim=keepdim) if dim is not None else input.sum()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(torch.ones(input_size))


@register_function("cumsum")
class AutogradCumsum(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim = input
        ctx.save_for_backward(dim)
        return input.cumsum(dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, = ctx.saved_tensors
        return grad_output.flip(dim).cumsum(dim).flip(dim)


@register_function("trace")
class AutogradTrace(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.size()[0])
        return input.trace()

    @staticmethod
    def backward(ctx, grad_output):
        size, = ctx.saved_tensors
        return grad_output.new(torch.eye(size)).mul_(grad_output)


@register_function("mean")
class AutogradMean(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.mean(dim, keepdim=keepdim) if dim is not None else input.mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors

        # Handle special case where input is 0-dimensional
        if len(input_size) == 0:
            return grad_output

        nelement = float(
            reduce(lambda x, y: x * y, input_size) if dim is None else input_size[dim]
        )
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(torch.ones(input_size)).div_(nelement)


@register_function("var")
class AutogradVariance(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input, dim, keepdim))
        return input.var(dim, keepdim=keepdim) if dim is not None else input.var()

    @staticmethod
    def backward(ctx, grad_output):
        input, dim, keepdim = ctx.saved_tensors
        nelement = float(
            reduce(lambda x, y: x * y, input.size()) if dim is None else input.size(dim)
        )
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        mean = input.mean() if dim is None else input.mean(dim=dim, keepdim=keepdim)
        return (input - mean).mul(2.0).mul(grad_output).div(nelement)


@register_function("min")
class AutogradMin(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):

        # find minimum value (and corresponding argmin):
        if dim is None:
            argmin = input.argmin(one_hot=True)
            min = input.mul(argmin).sum()
        else:
            min, argmin = input.min(dim=dim, keepdim=keepdim, one_hot=True)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmin))
        if dim is None:
            return min
        else:
            ctx.mark_non_differentiable(argmin)
            return min, argmin

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmin = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmin)


@register_function("max")
class AutogradMax(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):

        # find maximum value (and corresponding argmax):
        if dim is None:
            argmax = input.argmax(one_hot=True)
            max = input.mul(argmax).sum()
        else:
            max, argmax = input.max(dim=dim, keepdim=keepdim, one_hot=True)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmax))
        if dim is None:
            return max
        else:
            ctx.mark_non_differentiable(argmax)
            return max, argmax

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmax = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmax)


@register_function("sigmoid")
class AutogradSigmoid(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        probs = input.sigmoid()
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, = ctx.saved_tensors
        return grad_output.mul(probs).mul_(probs.neg().add_(1.0))


@register_function("softmax")
class AutogradSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim = input
        probs = input.softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        return grad_output.add(-probs.mul(grad_output).sum(dim, keepdim=True)).mul_(
            probs
        )


@register_function("log_softmax")
class AutogradLogSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        input, dim = input
        probs = input.log_softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        z = probs.exp()
        result = grad_output - z * grad_output.sum(dim, keepdim=True)
        return result


@register_function("pad")
class AutogradPad(AutogradFunction):
    @staticmethod
    def forward(ctx, input, value=0.0, mode="constant"):
        input, padding = input
        ctx.save_for_backward(padding)
        output = input.pad(padding, value=value, mode=mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        padding, = ctx.saved_tensors
        for idx in range(0, len(padding), 2):
            dim = grad_output.dim() - (idx // 2) - 1
            start = padding[idx]
            end = grad_output.size(dim) - padding[idx + 1] - padding[idx]
            grad_output = grad_output.narrow(dim, start, end)
        return grad_output


@register_function("avg_pool2d")
class AutogradAvgPool2D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, padding=0, stride=None):

        # preprocess inputs:
        input, kernel_size = input
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(kernel_size, (int, float)):
            kernel_size = (kernel_size, kernel_size)

        # perform average pooling:
        output = input.avg_pool2d(kernel_size, padding=padding, stride=stride)

        # store information for backward pass:
        ctx.save_multiple_for_backward(
            (input.shape, output, kernel_size, padding, stride)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the gradient with respect to the input"""
        input_shape, output, kernel_size, padding, stride = ctx.saved_tensors
        assert stride[0] == stride[1], "stride must be same in all axes"
        assert padding[0] == padding[1], "padding must be same in all axes"

        in_channels = input_shape[1]
        # compute as d conv2d / d input with kernel as average filter
        kernel = torch.ones(in_channels, 1, kernel_size[0], kernel_size[1]) / (
            kernel_size[0] * kernel_size[1]
        )

        grad_input_padding = torch.nn.grad._grad_input_padding(
            grad_output, input_shape, stride, padding, kernel_size
        )

        # set groups=in_channels so input gradient is computed per channel
        if isinstance(grad_output, crypten.CrypTensor):
            return grad_output.conv_transpose2d(
                kernel,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=grad_input_padding,
                groups=in_channels,
                dilation=1,
            )

        return torch.conv_transpose2d(
            grad_output,
            kernel,
            bias=None,
            stride=stride,
            padding=padding,
            output_padding=grad_input_padding,
            groups=in_channels,
            dilation=1,
        )


@register_function("max_pool2d")
class AutogradMaxPool2D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, padding=0, stride=None, return_indices=False):

        # preprocess inputs:
        input, kernel_size = input
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)

        # perform max pooling:
        # Note return_indices is required to be True to computing backward.
        output, indices = input.max_pool2d(
            kernel_size, padding=padding, stride=stride, return_indices=True
        )

        # store information for backward pass and return:
        ctx.save_multiple_for_backward(
            (input.size(), indices, kernel_size, padding, stride)
        )
        if return_indices:
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        output_size, indices, kernel_size, padding, stride = ctx.saved_tensors
        assert stride[0] == stride[1], "stride must be same in all axes"
        assert padding[0] == padding[1], "padding must be same in all axes"
        return grad_output._max_pool2d_backward(
            indices,
            kernel_size,
            padding=padding,
            stride=stride,
            output_size=output_size,
        )


@register_function("conv2d")
class AutogradConv2D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, padding=0, stride=1):
        input, kernel = input
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        ctx.save_multiple_for_backward((input, kernel, padding, stride))
        return input.conv2d(kernel, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):

        # get input, kernel, and sizes:
        input, kernel, padding, stride = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.size()
        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        # compute gradient with respect to input:
        output_padding = torch.nn.grad._grad_input_padding(
            grad_output, input.size(), stride, padding, (kernel_size_y, kernel_size_x)
        )
        grad_input = grad_output.conv_transpose2d(
            kernel, stride=stride, padding=padding, output_padding=output_padding
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels, 1, 1)
        grad_output = grad_output.view(
            grad_output.size(0) * grad_output.size(1),
            1,
            grad_output.size(2),
            grad_output.size(3),
        )
        input = input.view(
            1, input.size(0) * input.size(1), input.size(2), input.size(3)
        )
        # dilation is set to stride based on PyTorch's conv2d_weight implementation
        grad_kernel = input.conv2d(
            grad_output,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(
            batch_size,
            grad_kernel.size(1) // batch_size,
            grad_kernel.size(2),
            grad_kernel.size(3),
        )
        grad_kernel = (
            grad_kernel.sum(dim=0)
            .view(in_channels, out_channels, grad_kernel.size(2), grad_kernel.size(3))
            .transpose(0, 1)
        )
        grad_kernel = grad_kernel.narrow(2, 0, kernel_size_y)
        grad_kernel = grad_kernel.narrow(3, 0, kernel_size_x)
        return (grad_input, grad_kernel)


@register_function("batchnorm")
class AutogradBatchNorm(AutogradFunction):
    @staticmethod
    def forward(
        ctx,
        input,
        running_mean=None,
        running_var=None,
        training=False,
        eps=1e-05,
        momentum=0.1,
    ):
        """
        Computes forward step of batch norm by normalizing x
            and returning weight * x_norm + bias.

        Running mean and var are computed over the `C` dimension for an input
        of size `(N, C, +)`.

        Note: inv_var can introduce precision errors due to sqrt and division
            particularly when the number of samples in a batch is small.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context which
                stores parameters such as weight and bias for backward step.
            input (tuple of torch.tensors or cryptensor):
                containing (x, weight, bias) with shapes `(N, C, +)`, `C`, and `C`
                in turn.
            training (bool): if training is True, running mean and var are
                updated with the momentum factor and stored in module. Forward
                is performed using batch statistics. If training is False,
                running statistics are used and therefore cannot be none.
            running_mean (torch.tensor or cryptensor): with shape `C`
            running_var (torch.tensor or cryptensor): with shape `C`
            eps (float): specifies epsilon used for numerical precision in inv_var
            momentum (float): moment factor used in updating running mean and var.

        Returns: (weight * normalized input + bias) of shape `(N, C, +)`.
        """
        x, weight, bias = input

        # determine dimensions over which means and variances are computed
        # for an input of shape (N, C, +), stats are computed over the C dimension
        stats_dimensions = list(range(x.dim()))
        # remove C dimension
        stats_dimensions.pop(1)
        # shape for broadcasting stats with x
        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = x.shape[1]

        if training:
            # track batch statistics
            mean = x.mean(dim=stats_dimensions)
            variance = x.var(dim=stats_dimensions)

            if running_mean is not None and running_var is not None:
                running_var.set(running_var * (1.0 - momentum) + variance * momentum)
                running_mean.set(running_mean * (1.0 - momentum) + mean * momentum)
        else:
            if running_mean is None or running_var is None:
                raise ValueError(
                    "Must provide running_mean and running_var when training is False"
                )
            mean = running_mean
            variance = running_var

        if torch.is_tensor(variance):
            inv_var = 1 / torch.sqrt(variance + eps)
        else:
            inv_var = (variance + eps).pos_pow(-0.5)

        # reshape shape (C) to broadcastable (1, C, 1, +)
        mean = mean.reshape(broadcast_shape)
        inv_var = inv_var.reshape(broadcast_shape)
        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        x_norm = (x - mean) * inv_var

        ctx.save_multiple_for_backward((x_norm, weight, inv_var))
        return x_norm * weight + bias

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient with respect to x, weight, and bias.

        Statistics are assumed to be computed along dimension C
        for an input of shape (N, C, ...). Note, partials with respect to
        the input treat mean and variance as constants similar to torch.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context containing
                x_norm, weight, and inv_var. Note weight
                and inv_var must be broadcastable with grad_output.
            grad_output (cryptensor): batchnorm output of shape (N, C, +).

        Returns:
            x_grad (cryptensor): gradient with respect to x with shape (N, C, +).
            weight_grad (cryptensor): gradient with respect to the weight of
                with shape (C).
            bias_grad (cryptensor): gradient with respect to bias of shape (C).
        """
        x_norm, weight, inv_var = ctx.saved_tensors
        # determine dimensions over which means and variances are computed
        # statistics are computed over C dimension for input shape: (N, C, +)
        stats_dimensions = list(range(len(grad_output.shape)))
        stats_dimensions.pop(1)

        x_grad = grad_output * weight * inv_var

        weight_grad = grad_output.mul(x_norm)
        weight_grad = weight_grad.sum(dim=stats_dimensions)

        bias_grad = grad_output.clone()
        bias_grad = bias_grad.sum(dim=stats_dimensions)

        return (x_grad, weight_grad, bias_grad)


@register_function("binary_cross_entropy")
class AutogradBinaryCrossEntropy(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        pred, target = input
        ctx.save_multiple_for_backward([pred, target])
        ctx.mark_non_differentiable(target)
        log_pos, log_neg = crypten.stack([pred, 1.0 - pred]).log().unbind(dim=0)
        loss_values = target * log_pos + ((1.0 - target) * log_neg)
        return loss_values.sum().div(-target.nelement())

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        rec_pos, rec_neg = crypten.stack([pred, 1.0 - pred]).reciprocal().unbind(dim=0)
        grad = (rec_neg * (1.0 - target)) - rec_pos * target
        return grad.div_(target.nelement()).mul_(grad_output)


@register_function("cross_entropy")
class AutogradCrossEntropy(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        pred, target = input  # NOTE: target is assumed to be one-hot vector.
        softmax = pred.softmax(1)
        ctx.save_multiple_for_backward([softmax, target])
        ctx.mark_non_differentiable(target)
        loss_values = softmax.log().mul_(target).neg_()
        return loss_values.sum().div_(target.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors
        loss_grad = softmax.sub(target)
        return loss_grad.div_(target.size(0)).mul_(grad_output)
