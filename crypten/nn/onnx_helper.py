#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
from onnx import numpy_helper
from torch.onnx import OperatorExportTypes


def get_parameter_name(name):
    """
    Gets parameter name from parameter key.
    """
    return name[name.rfind(".") + 1 :]


def get_attribute_value(attr):
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
    else:
        raise ValueError("Unknown attribute type for attribute %s." % attr.name)


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
