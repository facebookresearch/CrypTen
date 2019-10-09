#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io

import onnx
import torch
from onnx import numpy_helper

from .loss import BCELoss, CrossEntropyLoss, L1Loss, MSELoss
from .module import (
    Add,
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Concat,
    Constant,
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    Conv2d,
    Flatten,
    Gather,
    GlobalAveragePool,
    Graph,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Reshape,
    Sequential,
    Shape,
    Squeeze,
    Sub,
    Unsqueeze,
    _BatchNorm,
    _ConstantPad,
    _Pool2d,
)
from .onnx_helper import get_attribute_value, get_parameter_name


# expose contents of package:
__all__ = [
    "MSELoss",
    "L1Loss",
    "BCELoss",
    "Add",
    "AvgPool2d",
    "_BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Concat",
    "Constant",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Conv2d",
    "CrossEntropyLoss",
    "Flatten",
    "Gather",
    "GlobalAveragePool",
    "Graph",
    "Linear",
    "MaxPool2d",
    "Module",
    "_Pool2d",
    "ReLU",
    "Reshape",
    "Sequential",
    "Shape",
    "Sub",
    "Squeeze",
    "Unsqueeze",
]

# mapping from ONNX to crypten.nn:
ONNX_TO_CRYPTEN = {
    "Add": Add,
    "AveragePool": AvgPool2d,
    "BatchNormalization": _BatchNorm,
    "Concat": Concat,
    "Conv": Conv2d,
    "Constant": Constant,
    "Flatten": Flatten,
    "Gather": Gather,
    "Gemm": Linear,
    "GlobalAveragePool": GlobalAveragePool,
    "MaxPool": MaxPool2d,
    "Pad": _ConstantPad,
    "Relu": ReLU,
    "Reshape": Reshape,
    "Shape": Shape,
    "Sub": Sub,
    "Squeeze": Squeeze,
    "Unsqueeze": Unsqueeze,
}


def from_pytorch(pytorch_model, dummy_input):
    """
    Static function that converts a PyTorch model into a CrypTen model.
    """

    # export model to ONX graph:
    f = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        f,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
    )
    f.seek(0)

    # construct CrypTen model:
    crypten_model = from_onnx(f)

    # make sure training / eval setting is copied:
    crypten_model.train(mode=pytorch_model.training)
    return crypten_model


def from_onnx(onnx_string_or_file):
    """
    Constructs a CrypTen model or module from an ONNX Protobuf string or file.
    """

    # if input is file, read string:
    if hasattr(onnx_string_or_file, "seek"):  # input is file-like
        onnx_string_or_file.seek(0)
        onnx_model = onnx.load(onnx_string_or_file)
    else:
        onnx_model = onnx.load_model_from_string(onnx_string_or_file)

    # create dict of all parameters, inputs, and outputs:
    all_parameters = {
        t.name: torch.from_numpy(numpy_helper.to_array(t))
        for t in onnx_model.graph.initializer
    }
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    input_names = [
        name for name in input_names if name not in all_parameters.keys()
    ]  # parameters are not inputs
    assert len(input_names) == 1, "number of inputs should be 1"
    assert len(output_names) == 1, "number of outputs should be 1"

    # create graph by looping over nodes:
    crypten_model = Graph(input_names[0], output_names[0])
    for node in onnx_model.graph.node:
        # get operator type:
        if node.op_type not in ONNX_TO_CRYPTEN:
            raise ValueError("CrypTen does not support op %s." % node.op_type)
        cls = ONNX_TO_CRYPTEN[node.op_type]

        # retrieve inputs, outputs, attributes, and parameters for this node:
        node_output_name = [name for name in node.output][0]
        node_input_names = [name for name in node.input]  # includes parameters
        parameters = {
            get_parameter_name(name): all_parameters[name]
            for name in node_input_names
            if name in all_parameters and name not in input_names
        }  # all the parameters for the current module
        node_input_names = [
            name
            for name in node_input_names
            if get_parameter_name(name) not in parameters
        ]
        attributes = {attr.name: get_attribute_value(attr) for attr in node.attribute}

        # add CrypTen module to graph:
        crypten_module = cls.from_onnx(parameters=parameters, attributes=attributes)
        crypten_model.add_module(node_output_name, crypten_module, node_input_names)

    # return model (or module when there is only one module):
    num_modules = len([_ for _ in crypten_model.modules()])
    if num_modules == 1:
        for crypten_module in crypten_model.modules():
            return crypten_module
    else:
        return crypten_model
