#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import collections
import io
import pickle

import torch


class RestrictedUnpickler(pickle.Unpickler):
    __SAFE_CLASSES = {}

    @classmethod
    def register_safe_class(cls, input_class):
        assert isinstance(input_class, type), "Cannot register %s type as safe" % type(
            input_class
        )
        classname = str(input_class).split("'")[1]
        cls.__SAFE_CLASSES[classname] = input_class

    def find_class(self, module, name):
        if module == "builtins":
            return getattr(builtins, name)
        if module == "collections":
            return getattr(collections, name)

        mods = module.split(".")
        if mods[0] == "torch":
            result = torch
            for mod in mods[1:]:
                result = getattr(result, mod)
            result = getattr(result, name)
        else:
            classname = f"{module}.{name}"
            if classname not in self.__SAFE_CLASSES.keys():
                raise ValueError(
                    f"Deserialization is restricted for pickled module {classname}"
                )

            result = self.__SAFE_CLASSES[classname]

        return result


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
