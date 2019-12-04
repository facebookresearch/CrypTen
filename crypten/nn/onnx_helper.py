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
            if function_key == "dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_dropout
            if function_key == "dropout_":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_dropout_
            if function_key == "feature_dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_feature_dropout
            if function_key == "feature_dropout_":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_feature_dropout_


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
def _onnx_crypten_dropout_(g, input, p, train):
    """
    This function converts PyTorch's in-place Dropout module to an in-place Dropout
    module in the ONNX model. The operator created is identical to the out-of-place
    Dropout module above, but uses the in-place version of the name.
    """
    r, _ = g.op("_Dropout_", input, ratio_f=p, outputs=2)
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


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_feature_dropout_(g, input, p, train):
    """
    This function converts PyTorch's DropoutNd module to a DropoutNd module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the DropoutNd module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore DropoutNd modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the DropoutNd module needs to be included in the
    CrypTen-specific conversion.
    """
    r, _ = g.op("_DropoutNd_", input, ratio_f=p, outputs=2)
    return r


def _trace_with_inplace(func, args, operator_export_type, return_outs=False):
    """
    Monkey-patched version of `_trace` function in `torch.onnx.utils`. The only
    change is that `_force_outplace` keyword argument is set to `False` here, instead
    of the `True` value used in `torch.onnx.utils.py`
    """
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args,)

    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        func, args, _force_outplace=False, _return_inputs_states=True
    )
    torch.onnx.utils.warn_on_static_input_change(inputs_states)

    trace_graph = torch.onnx.utils._optimize_graph(trace_graph, operator_export_type)
    if return_outs:
        return trace_graph, torch_out
    return trace_graph


def _trace_and_get_graph_from_model_with_inplace(model, args, training):
    """
    Monkey-patched version of `_trace_and_get_graph_from_model` function in
    `torch.onnx.utils`. The only change is that `_force_outplace` keyword argument
    is set to `False` here, instead of the `True` value used in `torch.onnx.utils.py`
    """
    orig_state_dict_keys = torch.onnx.utils._unique_state_dict(model).keys()

    with torch.onnx.utils.set_training(model, training):
        trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
            model, args, _force_outplace=False, _return_inputs_states=True
        )
        torch.onnx.utils.warn_on_static_input_change(inputs_states)

    if orig_state_dict_keys != torch.onnx.utils._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _run_symbolic_function_with_in_place(
    g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX
):
    """
    Monkey-patched version of `_run_symbolic_function` function in `torch.onnx.utils`.
    The only change is that trailing '_' is no longer removed from `ns_op_name` for
    the dropout function.
    """
    try:
        import torch
        from torch.onnx.symbolic_helper import (
            _export_onnx_opset_version as opset_version,
        )
        import torch.onnx.symbolic_registry as sym_registry

        sym_registry.register_version("", opset_version)
        if operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK:
            import torch.onnx.symbolic_caffe2

            torch.onnx.symbolic_caffe2.register_quantized_ops("caffe2", opset_version)

        ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")
        if n.kind().endswith("_"):
            if op_name not in ["dropout_", "feature_dropout_"]:
                op_name = op_name[:-1]

        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = sym_registry.is_registered_op(
                op_name, "", opset_version
            )
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = (
                operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            )
            if is_onnx_aten_export or (
                not is_exportable_aten_op and is_aten_fallback_export
            ):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return torch.onnx.utils._graph_at(
                    g, op_name, *inputs, aten=True, **attrs
                )

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn(
                        "ONNX export failed on ATen operator {} because "
                        "torch.onnx.symbolic_opset{}.{} does not exist".format(
                            op_name, opset_version, op_name
                        )
                    )
                op_fn = sym_registry.get_registered_op(op_name, "", opset_version)
                return op_fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant" and not n.mustBeNone():
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                if n.kindOf("value") == "s":
                    return g.op("Constant", value_s=n["value"])
                elif n.kindOf("value") == "is":
                    value = (
                        torch.stack([torch.tensor(v) for v in n["value"]])
                        if n["value"]
                        else []
                    )
                    return g.op("Constant", value_t=value)
                elif n.output().type().kind() == "DeviceObjType":
                    return None
                else:
                    raise RuntimeError(
                        "Unsupported prim::Constant kind: `{}`".format(
                            n.kindOf("value")
                        )
                    )
            elif (
                n.mustBeNone() or op_name == "ListConstruct" or op_name == "ListUnpack"
            ):
                # None is not an ONNX operator; keep it as None
                # let the exporter handle finally eliminating these

                # For ListConstruct/ListUnpack, it will be erased in the
                # ONNX peephole pass
                return None
            elif op_name == "Loop" or op_name == "If":
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                new_node = (
                    new_op_outputs[0].node()
                    if n.outputsSize() > 1
                    else new_op_outputs.node()
                )
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(
                        b, new_block, operator_export_type, env
                    )
                return new_op_outputs
            else:
                symbolic_name = "prim_" + op_name
                is_exportable = sym_registry.is_registered_op(
                    symbolic_name, "", opset_version
                )
                if not is_exportable:
                    warnings.warn(
                        "ONNX export failed on primitive operator {}".format(op_name)
                    )
                symbolic_fn = sym_registry.get_registered_op(
                    symbolic_name, "", opset_version
                )
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        elif ns == "quantized":
            domain = ""
            if operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK:
                domain = "caffe2"
            attrs = {k: n[k] for k in n.attributeNames()}

            if not sym_registry.is_registered_op(op_name, domain, opset_version):
                warnings.warn(
                    "ONNX export failed on quantized operator {}::{} because "
                    "torch.onnx.symbolic_opset{}.{} does not exist. ".format(
                        ns, op_name, opset_version, op_name
                    )
                )
            op_fn = sym_registry.get_registered_op(op_name, domain, opset_version)
            return op_fn(g, *inputs, **attrs)

        # custom ops
        elif sym_registry.is_registered_version(ns, opset_version):
            if not sym_registry.is_registered_op(op_name, ns, opset_version):
                warnings.warn(
                    "ONNX export failed on custom operator {}::{} because "
                    "torch.onnx.symbolic_opset{}.{} does not exist.".format(
                        ns, op_name, opset_version, op_name
                    )
                )
            symbolic_fn = sym_registry.get_registered_op(op_name, ns, opset_version)
            attrs = {k: n[k] for k in n.attributeNames()}
            return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn(
                "ONNX export failed on an operator with unrecognized namespace "
                "{}::{}; If you are trying to export a custom operator, "
                "make sure you registered it with the right domain and version."
                "Otherwise please report a bug".format(ns, op_name)
            )
            return None

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name),)
        raise
