#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def communicator():
    """Returns distributed communicator."""
    import os
    from .communicator import DistributedCommunicator

    # set default arguments for communicator:
    default_args = {
        "DISTRIBUTED_BACKEND": "gloo",
        "RENDEZVOUS": "file:///tmp/sharedfile",
        "WORLD_SIZE": 1,
        "RANK": 0,
    }
    for key, val in default_args.items():
        if key not in os.environ:
            os.environ[key] = str(val)

    # return communicator:
    return DistributedCommunicator()


# initialize communicator:
comm = communicator()

import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401

# other imports:
from .cryptensor import CrypTensor
from .mpc import ptype
from .multiprocessing_pdb import pdb


# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary

# expose classes and functions in package:
__all__ = ["CrypTensor", "pdb", "mpc", "nn"]


def print_communication_stats():
    comm.print_communication_stats()


def reset_communication_stats():
    comm.reset_communication_stats()


# Set backend
__SUPPORTRED_BACKENDS = [crypten.mpc]
__default_backend = __SUPPORTRED_BACKENDS[0]


def set_default_backend(new_default_backend):
    """Sets the default cryptensor backend (mpc, he)"""
    global __default_backend
    assert new_default_backend in __SUPPORTRED_BACKENDS, (
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
