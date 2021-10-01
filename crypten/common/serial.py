#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import collections
import difflib
import inspect
import io
import logging
import os
import pickle
import shutil
import struct
import sys
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager

import torch
from torch.serialization import (
    _check_seekable,
    _get_restore_location,
    _is_zipfile,
    _maybe_decode_ascii,
    _should_read_directly,
    storage_to_tensor_type,
)


def _safe_load_from_bytes(b):
    return _safe_legacy_load(io.BytesIO(b))


# Legacy code from torch._utils_internal
def get_source_lines_and_file(obj, error_msg=None):
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = f"Can't get source for {obj}."
        if error_msg:
            msg += "\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


class RestrictedUnpickler(pickle.Unpickler):
    __SAFE_CLASSES = {
        "builtins.set": builtins.set,
        "collections.OrderedDict": collections.OrderedDict,
        "torch.nn.modules.activation.LogSigmoid": torch.nn.modules.activation.LogSigmoid,
        "torch.nn.modules.activation.LogSoftmax": torch.nn.modules.activation.LogSoftmax,
        "torch.nn.modules.activation.ReLU": torch.nn.modules.activation.ReLU,
        "torch.nn.modules.activation.Sigmoid": torch.nn.modules.activation.Sigmoid,
        "torch.nn.modules.activation.Softmax": torch.nn.modules.activation.Softmax,
        "torch.nn.modules.batchnorm.BatchNorm1d": torch.nn.modules.batchnorm.BatchNorm1d,
        "torch.nn.modules.batchnorm.BatchNorm2d": torch.nn.modules.batchnorm.BatchNorm2d,
        "torch.nn.modules.batchnorm.BatchNorm3d": torch.nn.modules.batchnorm.BatchNorm3d,
        "torch.nn.modules.conv.Conv1d": torch.nn.modules.conv.Conv1d,
        "torch.nn.modules.conv.Conv2d": torch.nn.modules.conv.Conv2d,
        "torch.nn.modules.conv.ConvTranspose1d": torch.nn.modules.conv.ConvTranspose1d,
        "torch.nn.modules.conv.ConvTranspose2d": torch.nn.modules.conv.ConvTranspose2d,
        "torch.nn.modules.dropout.Dropout2d": torch.nn.modules.dropout.Dropout2d,
        "torch.nn.modules.dropout.Dropout3d": torch.nn.modules.dropout.Dropout3d,
        "torch.nn.modules.flatten.Flatten": torch.nn.modules.flatten.Flatten,
        "torch.nn.modules.linear.Linear": torch.nn.modules.linear.Linear,
        "torch.nn.modules.loss.BCELoss": torch.nn.modules.loss.BCELoss,
        "torch.nn.modules.loss.BCEWithLogitsLoss": torch.nn.modules.loss.BCEWithLogitsLoss,
        "torch.nn.modules.loss.CrossEntropyLoss": torch.nn.modules.loss.CrossEntropyLoss,
        "torch.nn.modules.loss.L1Loss": torch.nn.modules.loss.L1Loss,
        "torch.nn.modules.loss.MSELoss": torch.nn.modules.loss.MSELoss,
        "torch.nn.modules.pooling.AvgPool2d": torch.nn.modules.pooling.AvgPool2d,
        "torch.nn.modules.pooling.MaxPool2d": torch.nn.modules.pooling.MaxPool2d,
        "torch._utils._rebuild_parameter": torch._utils._rebuild_parameter,
        "torch._utils._rebuild_tensor_v2": torch._utils._rebuild_tensor_v2,
        "torch.storage._load_from_bytes": _safe_load_from_bytes,
        "torch.Size": torch.Size,
        "torch.BFloat16Storage": torch.BFloat16Storage,
        "torch.BoolStorage": torch.BoolStorage,
        "torch.CharStorage": torch.CharStorage,
        "torch.ComplexDoubleStorage": torch.ComplexDoubleStorage,
        "torch.ComplexFloatStorage": torch.ComplexFloatStorage,
        "torch.HalfStorage": torch.HalfStorage,
        "torch._C.HalfStorageBase": torch._C.HalfStorageBase,
        "torch.IntStorage": torch.HalfStorage,
        "torch.LongStorage": torch.LongStorage,
        "torch.QInt32Storage": torch.QInt32Storage,
        "torch._C.QInt32StorageBase": torch._C.QInt32StorageBase,
        "torch.QInt8Storage": torch.QInt8Storage,
        "torch._C.QInt8StorageBase": torch._C.QInt8StorageBase,
        "torch.QUInt8Storage": torch.QUInt8Storage,
        "torch.ShortStorage": torch.ShortStorage,
        "torch.storage._StorageBase": torch.storage._StorageBase,
        "torch.ByteStorage": torch.ByteStorage,
        "torch.DoubleStorage": torch.DoubleStorage,
        "torch.FloatStorage": torch.FloatStorage,
    }

    @classmethod
    def register_safe_class(cls, input_class):
        assert isinstance(input_class, type), "Cannot register %s type as safe" % type(
            input_class
        )
        classname = str(input_class).split("'")[1]
        logging.info(f"Registering {classname} class as safe for deserialization.")
        cls.__SAFE_CLASSES[classname] = input_class

    def find_class(self, module, name):
        classname = f"{module}.{name}"
        if classname not in self.__SAFE_CLASSES.keys():
            raise ValueError(
                f"Deserialization is restricted for pickled module {classname}"
            )
        return self.__SAFE_CLASSES[classname]


def register_safe_class(input_class):
    RestrictedUnpickler.register_safe_class(input_class)


def _assert_empty_ordered_dict(x):
    assert isinstance(x, collections.OrderedDict)
    assert len(x) == 0


def _check_hooks_are_valid(result, hook_name):
    if hasattr(result, hook_name):
        _assert_empty_ordered_dict(getattr(result, hook_name))
    if hasattr(result, "parameters"):
        for param in result.parameters():
            _assert_empty_ordered_dict(getattr(param, hook_name))
    if hasattr(result, "modules"):
        for module in result.modules():
            _assert_empty_ordered_dict(getattr(module, hook_name))


def restricted_loads(s):
    result = RestrictedUnpickler(io.BytesIO(s)).load()
    if torch.is_tensor(result) or isinstance(result, torch.nn.Module):
        _check_hooks_are_valid(result, "_backward_hooks")
    return result


class safe_pickle:
    Unpickler = RestrictedUnpickler
    @staticmethod
    def load(f):
        return RestrictedUnpickler(f).load()


def _safe_legacy_load(f):
    return torch.serialization._legacy_load(f, map_location=None, pickle_module=safe_pickle)
