#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
from collections import OrderedDict

import onnx
import torch
import torch.onnx.utils
from onnx import numpy_helper

from .loss import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss
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
    Conv1d,
    Conv2d,
    Dropout,
    Dropout2d,
    Dropout3d,
    DropoutNd,
    Exp,
    Flatten,
    Gather,
    GlobalAveragePool,
    Graph,
    Linear,
    LogSoftmax,
    MatMul,
    MaxPool2d,
    Mean,
    Module,
    ReLU,
    Reshape,
    Sequential,
    Shape,
    Softmax,
    Squeeze,
    Sub,
    Sum,
    Transpose,
    Unsqueeze,
    _BatchNorm,
    _ConstantPad,
    _Pool2d,
)
from .onnx_helper import (
    _sync_parameters,
    _update_onnx_symbolic_registry,
    get_attribute_value,
    get_parameter_name,
)


try:
    import tensorflow as tf  # noqa
    import tf2onnx

    TF_AND_TF2ONNX = True
except ImportError:
    TF_AND_TF2ONNX = False


# expose contents of package:
__all__ = [
    "MSELoss",
    "L1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
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
    "Conv1d",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Dropout2d",
    "Dropout3d",
    "DropoutNd",
    "Exp",
    "Flatten",
    "Gather",
    "GlobalAveragePool",
    "Graph",
    "Linear",
    "LogSoftmax",
    "MaxPool2d",
    "Module",
    "_Pool2d",
    "ReLU",
    "Mean",
    "Sum",
    "Reshape",
    "Sequential",
    "Shape",
    "Softmax",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
]

# mapping from ONNX to crypten.nn:
ONNX_TO_CRYPTEN = {
    "Add": Add,
    "AveragePool": AvgPool2d,
    "BatchNormalization": _BatchNorm,
    "Concat": Concat,
    "Constant": Constant,
    "Dropout": Dropout,
    "Dropout2d": Dropout2d,
    "Dropout3d": Dropout3d,
    "DropoutNd": DropoutNd,
    "Exp": Exp,
    "Flatten": Flatten,
    "Gather": Gather,
    "Gemm": Linear,
    "GlobalAveragePool": GlobalAveragePool,
    "LogSoftmax": LogSoftmax,
    "MatMul": MatMul,
    "MaxPool": MaxPool2d,
    "Pad": _ConstantPad,
    "Relu": ReLU,
    "ReduceMean": Mean,
    "ReduceSum": Sum,
    "Reshape": Reshape,
    "Shape": Shape,
    "Softmax": Softmax,
    "Squeeze": Squeeze,
    "Sub": Sub,
    "Transpose": Transpose,
    "Unsqueeze": Unsqueeze,
}


def from_pytorch(pytorch_model, dummy_input):
    """
    Static function that converts a PyTorch model into a CrypTen model.
    """
    # Exporting model to ONNX graph:
    # TODO: Currently export twice because the torch-to-ONNX symbolic registry
    # only gets created on the first call.

    # export first time so symbolic registry is created
    f = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        f,
        do_constant_folding=False,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
    )
    # update ONNX symbolic registry with CrypTen-specific functions
    _update_onnx_symbolic_registry()

    # export again so the graph is created with CrypTen-specific registry
    f = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        f,
        do_constant_folding=False,
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
        # retrieve inputs, outputs, attributes, and parameters for this node:
        node_output_name = list(node.output)[0]
        node_input_names = list(node.input)  # includes parameters

        # Create parameters: OrderedDict is required to figure out mapping
        # between complex names and ONNX arguments
        parameters = OrderedDict()
        orig_parameter_names = []
        # add in all the parameters for the current module
        for i, name in enumerate(node_input_names):
            if name in all_parameters and name not in input_names:
                key = get_parameter_name(name)
                # the following is necessary because tf2onnx names multiple parameters
                # identically if they have the same value
                if TF_AND_TF2ONNX:
                    # only modify if we already have the key in parameters
                    if key in parameters:
                        key = key + "_" + str(i)
                parameters[key] = all_parameters[name]
                orig_parameter_names.append(get_parameter_name(name))
        node_input_names = [
            name
            for name in node_input_names
            if get_parameter_name(name) not in orig_parameter_names
        ]
        attributes = {attr.name: get_attribute_value(attr) for attr in node.attribute}

        # get operator type:
        if node.op_type == "Conv":
            dims = len(attributes["kernel_shape"])
            if dims == 1:
                cls = Conv1d
            elif dims == 2:
                cls = Conv2d
            else:
                raise ValueError("CrypTen does not support op Conv%dd." % dims)
        else:
            if node.op_type not in ONNX_TO_CRYPTEN:
                raise ValueError("CrypTen does not support op %s." % node.op_type)
            cls = ONNX_TO_CRYPTEN[node.op_type]

        if TF_AND_TF2ONNX:
            # sync parameter names so that they become what CrypTen expects
            parameters = _sync_parameters(parameters, node.op_type)

        # add CrypTen module to graph:
        crypten_module = cls.from_onnx(parameters=parameters, attributes=attributes)
        crypten_model.add_module(node_output_name, crypten_module, node_input_names)

    # return model (or module when there is only one module):
    num_modules = len(list(crypten_model.modules()))
    if num_modules == 1:
        for crypten_module in crypten_model.modules():
            return crypten_module
    else:
        return crypten_model
