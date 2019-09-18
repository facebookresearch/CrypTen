#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch.nn
from crypten.autograd_cryptensor import AutogradCrypTensor


class Module:
    """
    Base Module class that mimics the torch.nn.Module class.
    """

    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.encrypted = False
        self.train()

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        """
        Constructs a CrypTen model from an ONNX Protobuf string or file.
        """
        raise NotImplementedError("Call this function on a Module type.")

    def forward(self, input):
        """Perform forward pass on model."""
        raise NotImplementedError("forward not implemented")

    def __call__(self, input):
        return self.forward(input)

    def train(self, mode=True):
        """Sets the module in the specified training mode."""
        for param in self.parameters():
            param.requires_grad = mode
        self.training = mode
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)

    def register_module(self, name, module):
        """Registers module in the container."""
        self._modules[name] = module

    def modules(self):
        """Returns iterator over modules."""
        for _, module in self.named_modules():
            yield module

    def named_modules(self):
        """Returns iterator over named modules (non-recursively)."""
        for name, module in self._modules.items():
            yield name, module

    def register_parameter(self, name, param, requires_grad=True):
        """Register parameter in the module."""
        if name in self._parameters or hasattr(self, name):
            raise ValueError("Parameter or field %s already exists." % name)
        if torch.is_tensor(param):  # unencrypted model
            param.requires_grad = requires_grad
            self._parameters[name] = param
        else:                       # encryped model
            self._parameters[name] = AutogradCrypTensor(
                param, requires_grad=requires_grad
            )
        setattr(self, name, param)

    def set_parameter(self, name, param):
        """Sets value of parameter in the module."""
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        self._parameters[name] = param
        setattr(self, name, param)

    def parameters(self, recurse=True):
        """Iterator over parameters."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, recurse=True):
        """Iterator over named parameters."""
        for name, param in self._parameters.items():
            yield name, param
        if recurse:
            for module in self.modules():
                yield from module.named_parameters(recurse=recurse)

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for param in self.parameters():
            param.grad = None

    def update_parameters(self, learning_rate):
        """Performs gradient step on parameters."""
        for param in self.parameters():
            param.tensor.sub_(param.grad.mul(learning_rate))

    def register_buffer(self, name, buffer):
        """
        Register buffer in the module. Buffers are encrypted like parameters but
        they are not updated by parameter updates.
        """
        if name in self._buffers or hasattr(self, name):
            raise ValueError("Buffer or field %s already exists." % name)
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def set_buffer(self, name, buffer):
        """Sets value of buffer in the module."""
        if name not in self._buffers or not hasattr(self, name):
            raise ValueError("Buffer %s does not exist." % name)
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def buffers(self, recurse=True):
        """Iterator over buffers."""
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, recurse=True):
        """Iterator over named buffers."""
        for name, buffer in self._buffers.items():
            yield name, buffer
        if recurse:
            for module in self.modules():
                yield from module.named_buffers(recurse=recurse)

    def _apply(self, fn):
        """Applies a function recursively on all modules."""
        fn(self)
        for module in self.modules():
            module._apply(fn)
        return self

    def encrypt(self, mode=True, src=0):
        """Encrypts the model."""
        if mode != self.encrypted:

            # encrypt / decrypt parameters:
            self.encrypted = mode
            for name, param in self.named_parameters(recurse=False):
                requires_grad = param.requires_grad
                if mode:  # encrypt parameter
                    self.set_parameter(name, AutogradCrypTensor(
                        crypten.cryptensor(param, **{'src': src}),
                        requires_grad=requires_grad,
                    ))
                else:     # decrypt parameter
                    self.set_parameter(name, param.get_plain_text())
                    self._parameters[name].requires_grad = requires_grad

            # encrypt / decrypt buffers:
            for name, buffer in self.named_buffers(recurse=False):
                if mode:  # encrypt buffer
                    self.set_buffer(name, AutogradCrypTensor(
                        crypten.cryptensor(param, **{'src': src}),
                        requires_grad=False,
                    ))
                else:     # decrypt buffer
                    self.set_buffer(name, buffer.get_plain_text())

            # apply encryption recursively:
            return self._apply(lambda m: m.encrypt(mode=mode, src=src))
        return self

    def decrypt(self):
        """Decrypts model."""
        return self.encrypt(mode=False)


class Container(Module):
    """
    Container allows distinguishing between individual modules and containers.
    """

    pass


class Graph(Container):
    """
    Acyclic graph of modules.

    The module maintains a dict of named modules and a graph structure stored in
    a dict where each key is a module name, and the associated value is a list
    of module names that provide the input into the module.
    """

    def __init__(self, input_name, output_name, modules=None, graph=None):
        super().__init__()
        self.input_name = input_name
        self.output_name = output_name
        self._graph = {}
        if modules is not None:
            self._modules = modules
        if graph is not None:
            self._graph = graph

    def add_module(self, name, module, input_names):
        assert name not in self._graph, "Module %s already exists." % name
        self.register_module(name, module)
        self._graph[name] = input_names

    def forward(self, input):

        # keep track of all values that have been computed:
        values = {self.input_name: input}
        computed = {key: False for key in self._graph.keys()}
        inputs_available = {
            key: [False for _ in range(len(value_list))]
            for key, value_list in self._graph.items()
        }

        def _mark_as_computed(name):
            """Marks a value as having been computed."""
            computed[name] = True
            for key, value_list in self._graph.items():
                if name in value_list:
                    inputs_available[key][value_list.index(name)] = True

        def _find_computable_node():
            """Find a node for which all inputs are available."""
            for key, inputs_available_list in inputs_available.items():
                if all(inputs_available_list) and not computed[key]:
                    return key
            return None

        # perform forward pass:
        _mark_as_computed(self.input_name)
        node_to_compute = _find_computable_node()
        while node_to_compute is not None:

            # compute and store output of module:
            input = [values[name] for name in self._graph[node_to_compute]]
            if len(input) == 1:
                input = input[0]  # unpack iterable if possible
            output = self._modules[node_to_compute](input)
            values[node_to_compute] = output
            _mark_as_computed(node_to_compute)

            # return output if it is available:
            if node_to_compute == self.output_name:
                return output

            # find next node to compute:
            node_to_compute = _find_computable_node()

        # this should never happen:
        raise ValueError("nn.Graph.forward() failed. Is graph unconnected?")


class Sequential(Graph):
    """
    Sequence of modules.
    """

    def __init__(self, module_list):
        super().__init__("input", "output")
        num_modules = len(module_list)
        for idx, module in enumerate(module_list):
            module_name = "output" if idx + 1 == num_modules else "module_%d" % idx
            input_name = "input" if idx == 0 else "module_%d" % (idx - 1)
            self.add_module(module_name, module, [input_name])


class Constant(Module):
    """
    Modules that returns a constant.
    """

    def __init__(self, value):
        super().__init__()
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        value = value.to(dtype=torch.float)
        self.register_parameter("value", value)

    def forward(self, input):
        return self.value

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        assert "value" in attributes, "No value for Constant specified."
        return Constant(attributes["value"])


class Add(Module):
    """
    Module that sums two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].add(input[1])

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return Add()


class Sub(Module):
    """
    Module that subtracts two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].sub(input[1])

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return Sub()


class Squeeze(Module):
    """
    Module that squeezes a tensor.
    """

    def __init__(self, dimension):
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        return input.squeeze(self.dimension)

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axes"]
        assert len(dimension) == 1, "can only squeeze one dimension at a time"
        return Squeeze(dimension[0])


class Unsqueeze(Module):
    """
    Module that unsqueezes a tensor.
    """

    def __init__(self, dimension):
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        return input.unsqueeze(self.dimension)

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axes"]
        assert len(dimension) == 1, "can only unsqueeze one dimension at a time"
        return Unsqueeze(dimension[0])


class Flatten(Module):
    """
    Module that flattens the input tensor into a 2D matrix.
    """

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        if self.axis == 0:
            return x.view(1, -1)
        else:
            assert self.axis <= x.dim(), "axis must not be larger than dimension"
            prod = 1
            for i in range(self.axis):
                prod *= x.size(i)
            return x.view(prod, -1)

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        # axis : int (default is 1)
        axis = 1
        if "axis" in attributes:
            axis = int(attributes["axis"])
            assert axis >= 0, "axis must not be negative"
        return Flatten(axis)


class Shape(Module):
    """
    Module that returns the shape of a tensor. If the input tensor is encrypted,
    the output size vector will be encrypted, too.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = torch.tensor(x.size())
        if crypten.is_encrypted_tensor(x):
            size = crypten.cryptensor(size.float())
        return size

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return Shape()


class Concat(Module):
    """
    Module that concatenates tensors along a dimension.
    """

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input needs to be a list or tuple"
        assert len(input) >= 1, "need at least one tensor to concatenate"
        return crypten.cat(input, self.dimension)

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axis"]
        return Concat(dimension)


class Reshape(Module):
    """
    Module that reshapes tensors to new dimensions.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        tensor, shape = input

        # shape is not data so we can get plain text
        if crypten.is_encrypted_tensor(shape):
            shape = shape.get_plain_text()
        return tensor.reshape(shape.long().tolist())

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return Reshape()


class Gather(Module):
    """
    Module that gathers elements from tensor according to indices.
    """

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        tensor, indices = input

        # indices are not data so we can get plain text:
        if crypten.is_encrypted_tensor(indices):
            indices = indices.get_plain_text().long()
        result = tensor.take(indices, self.dimension)
        return result

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        # FIXME(shobha): Add test for this code path.
        if attributes is None:
            attributes = {}
        return Gather(attributes["axis"])


class _ConstantPad(Module):
    """
    Module that pads a tensor.
    """

    def __init__(self, padding, value, mode="constant"):
        super().__init__()
        if isinstance(padding, (int)):
            padding = [padding]
        self.padding = padding
        self.value = value
        self.mode = mode

    def forward(self, input):
        return input.pad(self.padding, value=self.value, mode="constant")

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if attributes is None:
            attributes = {}
        return _ConstantPad(
            attributes["pads"], attributes["value"], mode=attributes["mode"]
        )


class ConstantPad1d(_ConstantPad):
    """
    Module that pads a 1D tensor.
    """

    pass


class ConstantPad2d(_ConstantPad):
    """
    Module that pads a 2D tensor.
    """

    pass


class ConstantPad3d(_ConstantPad):
    """
    Module that pads a 3D tensor.
    """

    pass


class Linear(Module):
    """
    Module that performs linear transformation.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # initialize model parameters:
        pytorch_module = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x.matmul(self.weight.t()) + self.bias

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        if parameters is None:
            parameters = {}

        # create module:
        in_features = parameters["weight"].size(1)
        out_features = parameters["weight"].size(0)
        module = Linear(in_features, out_features, bias=("bias" in parameters))

        # set parameters:
        for key, value in parameters.items():
            module.set_parameter(key, value)
        return module


class Conv2d(Module):
    """
    Module that performs 2D convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):

        # check inputs:
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert _all_the_same(kernel_size), "only square kernels are supported"
            kernel_size = kernel_size[0]
        if isinstance(stride, (list, tuple)):
            assert _all_the_same(stride), "stride must be the same in each dimension"
            stride = stride[0]
        if isinstance(padding, (list, tuple)):
            assert _all_the_same(padding), "padding must be the same in each dimension"
            padding = padding[0]

        # initialize model parameters:
        pytorch_module = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

        # set other instance fields:
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.conv2d(self.weight, stride=self.stride, padding=self.padding)
        if hasattr(self, "bias"):
            x = x.add(self.bias.unsqueeze(-1).unsqueeze(-1))
        return x

    @staticmethod
    def from_onnx(parameters=None, attributes=None):

        # check parameters and attributes:
        if parameters is None:
            parameters = {}
        if attributes is None:
            attributes = {}
        assert _all_the_same(["kernel_shape"]), "only square kernels are supported"
        assert _all_the_same(
            attributes["strides"]
        ), "stride must be the same in each dimension"
        assert _all_the_same(
            attributes["pads"]
        ), "padding must be the same in each dimension"
        assert attributes["group"] == 1, "group convolution not supported"
        assert all(
            dilation == 1 for dilation in attributes["dilations"]
        ), "dilated convolutions not supported"

        # initialize module:
        in_channels = parameters["weight"].size(1)
        out_channels = parameters["weight"].size(0)
        module = Conv2d(
            in_channels,
            out_channels,
            attributes["kernel_shape"][0],
            stride=attributes["strides"][0],
            padding=attributes["pads"][0],
            bias=("bias" in parameters),
        )

        # set parameters:
        for key, value in parameters.items():
            module.set_parameter(key, value)
        return module


class ReLU(Module):
    """
    Module that computes rectified linear unit (ReLU) activations.
    """

    def forward(self, x):
        return x.relu()

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return ReLU()


class _Pool2d(Module):
    """
    Module that performs 2D pooling.
    """

    def __init__(
        self, pool_type, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        args = [self.kernel_size]
        kwargs = {"stride": self.stride, "padding": self.padding}
        if self.pool_type == "average":
            return x.avg_pool2d(*args, **kwargs)
        elif self.pool_type == "max":
            return x.max_pool2d(*args, **kwargs)
        else:
            raise ValueError("Unknown pooling type: %s" % self.pool_type)

    @staticmethod
    def from_onnx(pool_type, parameters=None, attributes=None):

        # check attributes:
        if attributes is None:
            attributes = {}
        if "pads" not in attributes:
            attributes["pads"] = [0]
        assert _all_the_same(["kernel_shape"]), "only square kernels are supported"
        assert _all_the_same(
            attributes["strides"]
        ), "stride must be the same in each dimension"
        assert _all_the_same(
            attributes["pads"]
        ), "padding must be the same in each dimension"

        # initialize module
        args = [attributes["kernel_shape"][0]]
        kwargs = {"stride": attributes["strides"][0], "padding": attributes["pads"][0]}
        if pool_type == "average":
            return AvgPool2d(*args, **kwargs)
        elif pool_type == "max":
            return MaxPool2d(*args, **kwargs)
        else:
            raise ValueError("Unknown pooling type: %s" % pool_type)


class AvgPool2d(_Pool2d):
    """
    Module that performs 2D average pooling.
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__("average", kernel_size, stride=stride, padding=padding)

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return super(AvgPool2d, AvgPool2d).from_onnx(
            "average", parameters=parameters, attributes=attributes
        )


class MaxPool2d(_Pool2d):
    """
    Module that performs 2D max pooling.
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__(
            "max",
            kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return super(MaxPool2d, MaxPool2d).from_onnx(
            "max", parameters=parameters, attributes=attributes
        )


class GlobalAveragePool(Module):
    """
    Module that implements GlobalAveragePool of ONNX.
    """

    def forward(self, input):
        assert input.dim() > 2, "input needs to have more than two dimensions"

        # sum over all but batch dimension:
        result = input.shallow_copy()
        for dim in range(2, input.dim()):
            result = result.sum(dim=dim, keepdim=True)

        # return average value:
        first_two_dims = input.size(0) * input.size(1)
        return result.div(input.nelement() / float(first_two_dims))

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        return GlobalAveragePool()


class _BatchNorm(Module):
    """
    Module that performs batch normalization on 1D tensors.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()

        # initialize model parameters and buffers:
        pytorch_module = torch.nn.BatchNorm1d(num_features)
        for param in ["weight", "bias"]:
            self.register_parameter(param, getattr(pytorch_module, param))
        for buffer in ["running_mean", "running_var"]:
            self.register_buffer(buffer, getattr(pytorch_module, buffer))

        # set model attributes:
        self.eps = eps
        self.momentum = momentum

    def forward(self, input):
        return input.batchnorm(
            self.weight, self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum,
        )

    @staticmethod
    def from_onnx(parameters=None, attributes=None):

        # preprocess all attributes:
        if parameters is None:
            parameters = {}
        if attributes is None:
            attributes = {}
        num_features = len(parameters["running_mean"])

        # create module:
        kwargs = {"eps": attributes["epsilon"], "momentum": attributes["momentum"]}
        module = _BatchNorm(num_features, **kwargs)

        # set parameters:
        for key, value in parameters.items():
            if key in ["running_mean", "running_var"]:
                module.set_buffer(key, value)
            else:
                module.set_parameter(key, value)
        return module


class BatchNorm1d(_BatchNorm):
    """
    Module that performs batch normalization on 1D tensors.
    """

    pass


class BatchNorm2d(_BatchNorm):
    """
    Module that performs batch normalization on 2D tensors.
    """

    pass


class BatchNorm3d(_BatchNorm):
    """
    Module that performs batch normalization on 3D tensors.
    """

    pass


def _all_the_same(items):
    """
    Checks whether all values in a list are the same.
    """
    return all(items[0] == item for item in items)
