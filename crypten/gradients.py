#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
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
            raise ValueError("Function (%s: %s) must extend AutogradFunction" %
                             (name, cls.__name__))
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
        return grad_output.sum().unsqueeze(-1)

    # remove batch dimension:
    if grad_output.dim() == len(input_size) + 1:
        grad_output = grad_output.sum(0, keepdim=False)
    assert grad_output.dim() == len(input_size), \
        "cannot perform inverse broadcast"

    # perform accumulation across broadcast dimensions:
    for dim in range(grad_output.dim()):
        if input_size[dim] == 1 and grad_output.size(dim) > 1:
            grad_output = grad_output.sum(dim, keepdim=True)
    return grad_output


class AutogradFunction(object):
    """
    Base implementation of a function that supports autograd.
    """
    differentiable = True

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


@register_function("view")
class AutogradView(AutogradFunction):

    @staticmethod
    def forward(ctx, input):
        input, size = input
        ctx.save_for_backward(input)
        return input.view(size)

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


@register_function("squeeze")
class AutogradSqueeze(AutogradFunction):

    @staticmethod
    def forward(ctx, input):

        # preprocess inputs:
        if isinstance(input, (tuple, list)) and len(input) == 1:
            input, = input      # no dimension to squeeze specified
        else:
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
        mask = input.gt(0.)
        ctx.save_for_backward(mask)
        return input.mul(mask)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output.mul(mask)


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
        return (grad_output.matmul(other.t()), self_.t().matmul(grad_output))


@register_function("div")
class AutogradDiv(AutogradFunction):

    @staticmethod
    def forward(ctx, input):
        input, scalar = input
        ctx.save_for_backward(scalar)
        return input.div(scalar)

    @staticmethod
    def backward(ctx, grad_output):
        scalar, = ctx.saved_tensors
        return grad_output.div(scalar)  # may be sped up by caching input.log()


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


@register_function("square")
class AutogradSquare(AutogradFunction):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.square()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.mul(input.mul_(2.0))


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
        self_, other = input
        return (grad_output.matmul(other), grad_output.matmul(self_))


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
    def forward(ctx, input, dim=None, keepdim=False):
        norm = input.norm(dim=dim, keepdim=keepdim) if dim is not None \
            else input.norm()
        ctx.save_multiple_for_backward((input, norm, dim, keepdim))
        return norm

    @staticmethod
    def backward(ctx, grad_output):
        input, norm, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        return grad_output.mul(input.div(norm))


@register_function("sum")
class AutogradSum(AutogradFunction):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.sum(dim=dim, keepdim=keepdim) if dim is not None \
            else input.sum()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        return grad_output.new(torch.ones(input_size)).mul_(grad_output)


@register_function("mean")
class AutogradMean(AutogradFunction):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.mean(dim, keepdim=keepdim) if dim is not None \
            else input.mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        nelement = float(reduce(lambda x, y: x * y, input_size)
                         if dim is None else input_size[dim])
        return grad_output.new(torch.ones(input_size)).mul_(grad_output).div_(nelement)


@register_function("var")
class AutogradVariance(AutogradFunction):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.save_multiple_for_backward((input, dim, keepdim))
        return input.var(dim, keepdim=keepdim) if dim is not None \
            else input.var()

    @staticmethod
    def backward(ctx, grad_output):
        input, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        nelement = float(reduce(lambda x, y: x * y, input.size())
                         if dim is None else input.size(dim))
        mean = input.mean() if dim is None else input.mean(dim=dim, keepdim=keepdim)
        return (input - mean).mul_(2.0).mul_(grad_output).div_(nelement)


@register_function("min")
class AutogradMin(AutogradFunction):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):

        # find minimum value (and corresponding argmin):
        min = input.min(dim=dim, keepdim=keepdim, one_hot=True) \
            if dim is not None else input.min(one_hot=True)
        argmin = None
        if isinstance(min, tuple):
            min, argmin = min

        # save context and return:
        ctx.save_multiple_for_backward((input, dim, keepdim, argmin))
        return min

    @staticmethod
    def backward(ctx, grad_output):
        input, dim, keepdim, argmin = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        if argmin is None:
            argmin = input.argmin(one_hot=True)
        return grad_output.mul(argmin)


@register_function("max")
class AutogradMax(AutogradFunction):

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):

        # find maximum value (and corresponding argmax):
        max = input.max(dim=dim, keepdim=keepdim, one_hot=True) \
            if dim is not None else input.max(one_hot=True)
        argmax = None
        if isinstance(max, tuple):
            max, argmax = max

        # save context and return:
        ctx.save_multiple_for_backward((input, dim, keepdim, argmax))
        return max

    @staticmethod
    def backward(ctx, grad_output):
        input, dim, keepdim, argmax = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)
        if argmax is None:
            argmax = input.argmax(one_hot=True)
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
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, = ctx.saved_tensors
        return grad_output.add(-probs.mul(grad_output).sum(1, keepdim=True)).mul_(probs)


@register_function("pad")
class AutogradPad(AutogradFunction):

    @staticmethod
    def forward(ctx, input, value=0., mode="constant"):
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

        # perform average pooling:
        output = input.avg_pool2d(kernel_size, padding=padding, stride=stride)

        # store information for backward pass:
        ctx.save_multiple_for_backward((input, output, kernel_size, padding, stride))
        return output

    @staticmethod
    def backward(ctx, grad_output):

        # create average pooling kernel:
        input, output, kernel_size, padding, stride = ctx.saved_tensors
        inchannels = input.size(1)
        ones = torch.ones(inchannels, inchannels, kernel_size, kernel_size)

        # compute gradient with respect to input:
        output_padding = torch.nn.grad._grad_input_padding(
            grad_output, input.size(), stride, padding, (kernel_size, kernel_size)
        )
        return grad_output.conv_transpose2d(
            grad_output.new(ones.div_(float(kernel_size ** 2))),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )


@register_function("max_pool2d")
class AutogradMaxPool2D(AutogradFunction):

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

        # perform max pooling:
        output, indices = input.max_pool2d(
            kernel_size, padding=padding, stride=stride, return_indices=True,
        )

        # store information for backward pass:
        ctx.save_multiple_for_backward(
            (input.size(), indices, kernel_size, padding, stride)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output_size, indices, kernel_size, padding, stride = ctx.saved_tensors
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
            kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels, 1, 1)
        grad_output = grad_output.view(
            grad_output.size(0) * grad_output.size(1),
            1,
            grad_output.size(2),
            grad_output.size(3),
        )
        input = input.view(1, input.size(0) * input.size(1),
                              input.size(2), input.size(3))
        grad_kernel = input.conv2d(
            grad_output,
            padding=padding,
            stride=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(
            batch_size,
            grad_kernel.size(1) // batch_size,
            grad_kernel.size(2),
            grad_kernel.size(3),
        )
        grad_kernel = grad_kernel.sum(dim=0).view(
            in_channels,
            out_channels,
            grad_kernel.size(2),
            grad_kernel.size(3),
        ).transpose(0, 1)
        grad_kernel = grad_kernel.narrow(2, 0, kernel_size_y)
        grad_kernel = grad_kernel.narrow(3, 0, kernel_size_x)
        return (grad_input, grad_kernel)


@register_function("batchnorm")
class AutogradBatchNorm(AutogradFunction):

    @staticmethod
    def forward(ctx, input, running_mean=None, running_var=None,
                training=False, eps=1e-05, momentum=0.1):

        # unpack inputs:
        input, weight, bias = input

        # determine dimensions over which means and variances are computed:
        dimensions = []
        if input.dim() == 3:    # 1D input
            dimensions = [0, 2]
        elif input.dim() == 4:  # 2D input
            dimensions = [0, 2, 3]
        elif input.dim() == 5:  # 3D input
            dimensions = [0, 2, 3, 4]

        # track batch statistics:
        if training:
            mean = input.mean(dimensions)
            variance = input.var(dimensions)
            if running_mean is not None and running_var is not None:
                pass
                # TODO: Add CrypTensor.set() method for this to work.
                # running_mean.set(running_mean.mul(1.0 - momentum)
                #                              .add(mean.mul(momentum)))
                # running_var.set(running_var.mul(1.0 - momentum)
                #                            .add(variance.mul(momentum)))
        else:
            mean = running_mean
            variance = running_var

        # compute bias and gain:
        inv_var = (variance + eps).pow(-0.5)
        alpha = inv_var * weight
        beta = bias - mean * alpha

        # ensure dimensionality of bias and gain matches input dimensionality:
        for dimension in dimensions:
            inv_var = inv_var.unsqueeze(dimension)
            alpha = alpha.unsqueeze(dimension)
            beta = beta.unsqueeze(dimension)

        # apply bias and gain:
        ctx.save_multiple_for_backward((input, alpha, inv_var, alpha.size()))
        return alpha * input + beta

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, inv_var, weight_size = ctx.saved_tensors
        weight_grad = grad_output.mul(input).mul(inv_var)
        bias_grad = grad_output.clone()
        return (
            grad_output.mul(alpha),
            _inverse_broadcast(weight_grad, weight_size).squeeze(),
            _inverse_broadcast(bias_grad, weight_size).squeeze(),
        )  # FIXME: Unit tests claim gradients are incorrect.
