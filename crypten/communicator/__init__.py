#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .communicator import Communicator
from .distributed_communicator import DistributedCommunicator


__comm = None
__is_initialized = False


def get():
    if not __is_initialized:
        raise RuntimeError("Crypten not initialized. Please call crypten.init() first.")
    return __comm


def init():
    global __is_initialized, __comm
    if __is_initialized:
        return

    __is_initialized = True

    import os

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

    __comm = DistributedCommunicator()


def uninit():
    global __comm, __is_initialized
    if __comm:
        __comm.shutdown()
    __comm = None
    __is_initialized = False


# expose classes and functions in package:
__all__ = ["Communicator", "DistributedCommunicator", "init", "uninit", "get"]
