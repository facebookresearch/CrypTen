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


# Adapt torch.load to use RestrictedUnpickler - patched for torch.storage._load_from_bytes
# (Adapted from https://github.com/pytorch/pytorch/blob/master/torch/serialization.py#L602-L773)
class SourceChangeWarning(Warning):
    pass


@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def _safe_legacy_load(f):
    MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
    PROTOCOL_VERSION = 1001
    deserialized_objects = {}

    restore_location = _get_restore_location(None)

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading."
            )
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + ".patch"
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise IOError
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except IOError:
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            msg = "source code of class '{container_type}' has changed. {msg}".format(
                container_type=torch.typename(container_type), msg=msg
            )
            warnings.warn(msg, SourceChangeWarning)

    def legacy_load(f):
        deserialized_objects = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(
            tarfile.open(fileobj=f, mode="r:", format=tarfile.PAX_FORMAT)
        ) as tar, mkdtemp() as tmpdir:

            tar.extract("storages", path=tmpdir)
            with open(os.path.join(tmpdir, "storages"), "rb", 0) as f:
                num_storages = RestrictedUnpickler(f).load()
                for _ in range(num_storages):
                    args = RestrictedUnpickler(f).load()
                    key, location, storage_type = args
                    obj = storage_type._new_with_file(f)
                    obj = restore_location(obj, location)
                    deserialized_objects[key] = obj

                storage_views = RestrictedUnpickler(f).load()
                for target_cdata, root_cdata, offset, size in storage_views:
                    root = deserialized_objects[root_cdata]
                    deserialized_objects[target_cdata] = root[offset : offset + size]

            tar.extract("tensors", path=tmpdir)
            with open(os.path.join(tmpdir, "tensors"), "rb", 0) as f:
                num_tensors = RestrictedUnpickler(f).load()
                for _ in range(num_tensors):
                    args = RestrictedUnpickler(f).load()
                    key, storage_id, original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    tensor_type = storage_to_tensor_type(storage)
                    (ndim,) = struct.unpack("<i", f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    size = struct.unpack("<{}q".format(ndim), f.read(8 * ndim))
                    stride = struct.unpack("<{}q".format(ndim), f.read(8 * ndim))
                    (storage_offset,) = struct.unpack("<q", f.read(8))
                    tensor = tensor_type().set_(storage, storage_offset, size, stride)
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile("pickle")
            unpickler = RestrictedUnpickler(pickle_file)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == "storage":
            data_type, root_key, location, size, view_metadata = data
            location = _maybe_decode_ascii(location)
            if root_key not in deserialized_objects:
                obj = data_type(size)
                obj._torch_load_uninitialized = True
                deserialized_objects[root_key] = restore_location(obj, location)
            storage = deserialized_objects[root_key]
            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = storage[
                        offset : offset + view_size
                    ]
                return deserialized_objects[view_key]
            else:
                return storage
        else:
            raise RuntimeError("Unknown saved id type: %s" % saved_id[0])

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                )
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    if not hasattr(f, "readinto") and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        raise RuntimeError(
            "torch.load does not work with file-like objects that do not implement"
            "readinto on Python 3.8.0 and 3.8.1. Received object of type"
            '"{}". Please update to Python 3.8.2 or newer to restore this'
            "functionality.".format(type(f))
        )

    magic_number = RestrictedUnpickler(f).load()
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = RestrictedUnpickler(f).load()
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("Invalid protocol version: %s" % protocol_version)

    _ = RestrictedUnpickler(f).load()  # _sys_info
    unpickler = RestrictedUnpickler(f)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = RestrictedUnpickler(f).load()

    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        deserialized_objects[key]._set_from_file(f, offset, f_should_read_directly)
        if offset is not None:
            offset = f.tell()

    return result
