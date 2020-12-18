#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .communicator import Communicator
from .distributed_communicator import DistributedCommunicator
from .in_process_communicator import InProcessCommunicator


__use_threads = False


def get():
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    if not cls.is_initialized():
        raise RuntimeError("Crypten not initialized. Please call crypten.init() first.")

    return cls.get()


def _init(use_threads, rank=0, world_size=1, init_ttp=False):
    global __tls, __use_threads
    __use_threads = use_threads
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator

    if cls.is_initialized():
        return

    cls.initialize(rank, world_size, init_ttp=init_ttp)


def uninit():
    global __use_threads
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls.shutdown()
    __use_threads = False


def is_initialized():
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    return cls.is_initialized()


# expose classes and functions in package:
__all__ = ["Communicator", "DistributedCommunicator", "get", "uninit", "is_initialized"]
