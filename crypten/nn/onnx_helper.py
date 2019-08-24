#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from onnx import numpy_helper


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
