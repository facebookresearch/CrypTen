#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import io
from collections import OrderedDict

import onnx
import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
import torch.onnx.utils
from onnx import numpy_helper

from . import module


try:
    import tensorflow as tf  # noqa
    import tf2onnx

    TF_AND_TF2ONNX = True
except ImportError:
    TF_AND_TF2ONNX = False


def from_pytorch(pytorch_model, dummy_input):
    """
    Static function that converts a PyTorch model into a CrypTen model.
    """
    # construct CrypTen model:
    f = _from_pytorch_to_bytes(pytorch_model, dummy_input)
    crypten_model = from_onnx(f)
    f.close()

    # make sure training / eval setting is copied:
    crypten_model.train(mode=pytorch_model.training)
    return crypten_model


def _from_pytorch_to_bytes(pytorch_model, dummy_input):
    """Returns I/O stream containing onnx graph with crypten specific ops"""
    # TODO: Currently export twice because the torch-to-ONNX symbolic registry
    # only gets created on the first call.
    with io.BytesIO() as f:
        _export_pytorch_model(f, pytorch_model, dummy_input)

    # update ONNX symbolic registry with CrypTen-specific functions
    _update_onnx_symbolic_registry()

    # export again so the graph is created with CrypTen-specific registry
    f = io.BytesIO()
    f = _export_pytorch_model(f, pytorch_model, dummy_input)
    f.seek(0)
    return f


def _export_pytorch_model(f, pytorch_model, dummy_input):
    """Returns a Binary I/O stream containing exported model"""
    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "enable_onnx_checker": False,
        "input_names": ["input"],
        "output_names": ["output"],
    }
    try:
        # current version of PyTorch requires us to use `enable_onnx_checker`
        torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    except TypeError:
        # older versions of PyTorch require us to NOT use `enable_onnx_checker`
        kwargs.pop("enable_onnx_checker")
        torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    return f


def from_tensorflow(tensorflow_graph_def, inputs, outputs):
    """
    Static function that converts Tensorflow model into CrypTen model based on
    https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py
    The model is returned in evaluation mode.
    Args:
        `tensorflow_graph_def`: Input Tensorflow GraphDef to be converted
        `inputs`: input nodes
        `outputs`: output nodes
    """
    # Exporting model to ONNX graph
    if not TF_AND_TF2ONNX:
        raise ImportError("Please install both tensorflow and tf2onnx packages")

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tensorflow_graph_def, name="")
    with tf2onnx.tf_loader.tf_session(graph=tf_graph):
        g = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            opset=10,
            continue_on_error=False,
            input_names=inputs,
            output_names=outputs,
        )
    onnx_graph = tf2onnx.optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model(
        "converted from {}".format(tensorflow_graph_def)
    )
    f = io.BytesIO()
    f.write(model_proto.SerializeToString())

    # construct CrypTen model
    # Note: We don't convert crypten model to training mode, as Tensorflow
    # models are used for both training and evaluation without the specific
    # conversion of one mode to another
    f.seek(0)
    crypten_model = from_onnx(f)
    return crypten_model


def from_onnx(onnx_string_or_file):
    """Converts an onnx model to a CrypTen model"""
    converter = FromOnnx(onnx_string_or_file)
    crypten_model = converter.to_crypten()
    return crypten_model


class FromOnnx:
    """Converts Onnx Model to a CrypTen Model"""

    # mapping from ONNX to crypten.nn for modules with different names:
    ONNX_TO_CRYPTEN = {
        "adaptive_avg_pool2d": module.AdaptiveAvgPool2d,
        "adaptive_max_pool2d": module.AdaptiveMaxPool2d,
        "AveragePool": module.AvgPool2d,
        "BatchNormalization": module._BatchNorm,
        "Gemm": module.Linear,
        "MaxPool": module.MaxPool2d,
        "Pad": module._ConstantPad,
        "Relu": module.ReLU,
        "ReduceMean": module.Mean,
        "ReduceSum": module.Sum,
    }

    def __init__(self, onnx_string_or_file):
        onnx_model = FromOnnx._load_onnx_model(onnx_string_or_file)
        self.onnx_model = onnx_model

        self.all_parameters = {
            t.name: torch.from_numpy(numpy_helper.to_array(t))
            for t in onnx_model.graph.initializer
        }

    def to_crypten(self):
        """Constructs a CrypTen model from the onnx graph"""
        input_names, output_names = self._get_input_output_names()

        crypten_model = module.Graph(input_names[0], output_names[0])

        constant_module = None
        for node in self.onnx_model.graph.node:
            attributes = FromOnnx.get_attributes(node)
            parameters, node_input_names = self.get_parameters(node, input_names)

            crypten_class = self._get_operator_class(node.op_type, attributes)

            # Get shape from Constant graph input for classes that require a shape
            reshape_classes = [
                module.AdaptiveAvgPool2d,
                module.AdaptiveMaxPool2d,
                module.Reshape,
            ]
            if crypten_class in reshape_classes:
                assert (
                    constant_module is not None
                ), f"Pattern not supported: expected Constant shape before {crypten_class} node."
                attributes["shape"] = constant_module[1].value.long().tolist()
                constant_module = None

            if TF_AND_TF2ONNX:
                parameters = _sync_tensorflow_parameters(parameters, node.op_type)

            # add CrypTen module to graph
            crypten_module = crypten_class.from_onnx(
                parameters=parameters, attributes=attributes
            )
            node_output_name = list(node.output)[0]

            # Check if Constant is shape used for next node before adding Constant to module list
            if crypten_class.__name__ == "Constant":
                constant_module = (node_output_name, crypten_module, node_input_names)
            else:
                # Add Constant modules that are not shape inputs to graph
                if constant_module is not None:
                    crypten_model.add_module(*constant_module)
                    constant_module = None

                # Add CrypTen module to graph
                crypten_model.add_module(
                    node_output_name, crypten_module, node_input_names
                )

        crypten_model = FromOnnx._get_model_or_module(crypten_model)
        return crypten_model

    @staticmethod
    def _load_onnx_model(onnx_string_or_file):
        """Loads onnx model from file or string"""
        # if input is file, read string
        if hasattr(onnx_string_or_file, "seek"):
            onnx_string_or_file.seek(0)
            return onnx.load(onnx_string_or_file)
        return onnx.load_model_from_string(onnx_string_or_file)

    def _get_input_output_names(self):
        """Return input and output names"""
        input_names = []
        for input in self.onnx_model.graph.input:
            # parameters are not inputs
            if input.name not in self.all_parameters:
                input_names.append(input.name)

        output_names = [output.name for output in self.onnx_model.graph.output]

        assert len(input_names) == 1, "number of inputs should be 1"
        assert len(output_names) == 1, "number of outputs should be 1"

        return input_names, output_names

    def get_parameters(self, node, input_names):
        """Returns parameters (Ordered Dict) and node_input_names (list of str)"""
        # includes parameters
        node_input_names = list(node.input)

        # Create parameters: OrderedDict is required to figure out mapping
        # between complex names and ONNX arguments
        parameters = OrderedDict()
        orig_parameter_names = []

        # add in all the parameters for the current module
        for i, name in enumerate(node_input_names):
            if name in self.all_parameters and name not in input_names:
                key = FromOnnx._get_parameter_name(name)
                # the following is necessary because tf2onnx names multiple parameters
                # identically if they have the same value
                # only modify if we already have the key in parameters
                if TF_AND_TF2ONNX and key in parameters:
                    key = key + "_" + str(i)
                parameters[key] = self.all_parameters[name]
                orig_parameter_names.append(FromOnnx._get_parameter_name(name))
        node_input_names = [
            name
            for name in node_input_names
            if FromOnnx._get_parameter_name(name) not in orig_parameter_names
        ]
        return parameters, node_input_names

    @staticmethod
    def _get_model_or_module(crypten_model):
        """
        Returns module if model contains only one module. Otherwise returns model.
        """
        num_modules = len(list(crypten_model.modules()))
        if num_modules == 1:
            for crypten_module in crypten_model.modules():
                return crypten_module
        return crypten_model

    @staticmethod
    def _get_parameter_name(name):
        """
        Gets parameter name from parameter key.
        """
        return name[name.rfind(".") + 1 :]

    @staticmethod
    def get_attributes(node):
        attributes = {
            attr.name: FromOnnx._get_attribute_value(attr) for attr in node.attribute
        }
        return attributes

    @staticmethod
    def _get_attribute_value(attr):
        """
        Retrieves value from attribute in ONNX graph.
        """
        if attr.HasField("f"):  # floating-point attribute
            return attr.f
        elif attr.HasField("i"):  # integer attribute
            return attr.i
        elif attr.HasField("s"):  # string attribute
            return attr.s  # TODO: Sanitize string.
        elif attr.HasField("t"):  # tensor attribute
            return torch.from_numpy(numpy_helper.to_array(attr.t))
        elif len(attr.ints) > 0:
            return list(attr.ints)
        elif len(attr.floats) > 0:
            return list(attr.floats)
        raise ValueError("Unknown attribute type for attribute %s." % attr.name)

    @classmethod
    def _get_operator_class(cls, node_op_type, attributes):
        """Returns CrypTen class of operator"""
        # get operator type:
        if node_op_type == "Conv":
            dims = len(attributes["kernel_shape"])
            if dims == 1:
                crypten_class = module.Conv1d
            elif dims == 2:
                crypten_class = module.Conv2d
            else:
                raise ValueError("CrypTen does not support op Conv%dd." % dims)
        else:
            crypten_module = getattr(
                module, node_op_type, cls.ONNX_TO_CRYPTEN.get(node_op_type, None)
            )

            if crypten_module is None:
                raise ValueError("CrypTen does not support op %s." % node_op_type)
            crypten_class = crypten_module

        return crypten_class


def _sync_tensorflow_parameters(parameter_map, module_name):
    """
    Syncs parameters from parameter map to be consistent
    with expected PyTorch parameter map
    """

    def _map_module_parameters(parameter_map, module_param_names):
        for i, key in enumerate(parameter_map.keys()):
            value = parameter_map[key]
            new_parameter_map[module_param_names[i]] = value

    new_parameter_map = {}
    if module_name == "Conv":
        module_param_names = ["weight", "bias"]
        _map_module_parameters(parameter_map, module_param_names)
    elif module_name == "BatchNormalization":
        module_param_names = [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "training_mode",
        ]
        _map_module_parameters(parameter_map, module_param_names)
    else:
        new_parameter_map = parameter_map
    return new_parameter_map


def _update_onnx_symbolic_registry():
    """
    Updates the ONNX symbolic registry for operators that need a CrypTen-specific
    implementation and custom operators.
    """
    for version_key, version_val in sym_registry._registry.items():
        for function_key in version_val.keys():
            if function_key == "softmax":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_softmax
            if function_key == "log_softmax":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_logsoftmax
            if function_key == "dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_dropout
            if function_key == "feature_dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_feature_dropout


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_softmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's Softmax module to a Softmax module in
    the ONNX model. It overrides PyTorch's default conversion of Softmax module
    to a sequence of Exp, ReduceSum and Div modules, since this default
    conversion can cause numerical overflow when applied to CrypTensors.
    """
    result = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_logsoftmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's LogSoftmax module to a LogSoftmax module in
    the ONNX model. It overrides PyTorch's default conversion of LogSoftmax module
    to avoid potentially creating Transpose operators.
    """
    result = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_dropout(g, input, p, train):
    """
    This function converts PyTorch's Dropout module to a Dropout module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the Dropout module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore Dropout modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the Dropout module needs to be included in the
    CrypTen-specific conversion.
    """
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_feature_dropout(g, input, p, train):
    """
    This function converts PyTorch's DropoutNd module to a DropoutNd module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the DropoutNd module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore DropoutNd modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the DropoutNd module needs to be included in the
    CrypTen-specific conversion.
    """
    r, _ = g.op("DropoutNd", input, ratio_f=p, outputs=2)
    return r
