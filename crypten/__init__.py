#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator


def init():
    return crypten.communicator.init()


def uninit():
    return crypten.communicator.uninit()


import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401

# other imports:
from .cryptensor import CrypTensor
from .mpc import ptype


# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary


def print_communication_stats():
    crypten.communicator.get().print_communication_stats()


def reset_communication_stats():
    crypten.communicator.get().reset_communication_stats()


# Set backend
__SUPPORTED_BACKENDS = [crypten.mpc]
__default_backend = __SUPPORTED_BACKENDS[0]


def set_default_backend(new_default_backend):
    """Sets the default cryptensor backend (mpc, he)"""
    global __default_backend
    assert new_default_backend in __SUPPORTED_BACKENDS, (
        "Backend %s is not supported" % new_default_backend
    )
    __default_backend = new_default_backend


def get_default_backend():
    """Returns the default cryptensor backend (mpc, he)"""
    return __default_backend


def cryptensor(*args, backend=None, **kwargs):
    """
    Factory function to return encrypted tensor of given backend.
    """
    if backend is None:
        backend = get_default_backend()
    if backend == crypten.mpc:
        return backend.MPCTensor(*args, **kwargs)
    else:
        raise TypeError("Backend %s is not supported" % backend)


def is_encrypted_tensor(obj):
    """
    Returns True if obj is an encrypted tensor.
    """
    return isinstance(obj, CrypTensor)


# Top level tensor functions
__PASSTHROUGH_FUNCTIONS = ["bernoulli", "cat", "rand", "randperm", "stack"]


def __add_top_level_function(func_name):
    def _passthrough_function(*args, backend=None, **kwargs):
        if backend is None:
            backend = get_default_backend()
        return getattr(backend, func_name)(*args, **kwargs)

    globals()[func_name] = _passthrough_function


for func in __PASSTHROUGH_FUNCTIONS:
    __add_top_level_function(func)

# expose classes and functions in package:
__all__ = [
    "CrypTensor",
    "debug",
    "init",
    "mpc",
    "nn",
    "uninit",
]
