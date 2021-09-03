#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import sys
import tempfile
import traceback
import unittest
import warnings
from functools import wraps

import crypten.communicator as comm
import crypten.debug
import torch
import torch.distributed as dist
from crypten.config import cfg


def get_random_test_tensor(
    max_value=6, min_value=None, size=(1, 5), is_float=False, ex_zero=False, device=None
):
    """Generates random tensor for testing

    Args:
        max_value (int): defines maximum value for int tensor
        min_value (int): defines minimum value for int tensor
        size (tuple): size of tensor
        is_float (bool): determines float or int tensor
        ex_zero (bool): excludes zero tensor

    Returns: torch.tensor
    """
    if min_value is None:
        min_value = -max_value
    if is_float:
        tensor = (
            torch.rand(torch.Size(size), device=device) * (max_value - min_value)
            + min_value
        )
    else:
        tensor = torch.randint(
            min_value, max_value, torch.Size(size), dtype=torch.int64, device=device
        )
    if ex_zero:
        # replace 0 with 1
        tensor[tensor == 0] = 1

    # Broadcast this tensor to the world so that the generated random tensor
    # is in sync in all distributed processes. See T45688819 for more
    # information.
    tensor = comm.get().broadcast(tensor, 0)

    return tensor


def onehot(indices, num_targets=None):
    """
    Converts index vector into one-hot matrix.
    """
    assert indices.dtype == torch.long, "indices must be long integers"
    assert indices.min() >= 0, "indices must be non-negative"
    if num_targets is None:
        num_targets = indices.max() + 1
    onehot_vector = torch.zeros(indices.nelement(), num_targets, dtype=torch.long)
    onehot_vector.scatter_(1, indices.view(indices.nelement(), 1), 1)
    return onehot_vector


def get_random_linear(in_channels, out_channels):
    linear = torch.nn.Linear(in_channels, out_channels)
    if dist.is_initialized():
        # Broadcast this tensor to the world so that the generated random tensor
        # is in sync in all distributed processes. See T45688819 for more
        # information.
        comm.get().broadcast(linear.weight, 0)
        comm.get().broadcast(linear.bias, 0)

    return linear


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK = -1
    DEFAULT_DEVICE = "cpu"
    DEFAULT_WORLD_SIZE = 2

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn(self)

        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def __init__(self, methodName):
        super().__init__(methodName)

        self.device = torch.device(self.DEFAULT_DEVICE)
        self.rank = self.MAIN_PROCESS_RANK
        self.mp_context = multiprocessing.get_context("spawn")

    def setUp(self, world_size=DEFAULT_WORLD_SIZE):
        super(MultiProcessTestCase, self).setUp()

        crypten.debug.configure_logging()

        self.world_size = world_size
        self.default_tolerance = 0.5
        self.queue = self.mp_context.Queue()

        # This gets called in the children process as well to give subclasses a
        # chance to initialize themselves in the new process
        if self.rank == self.MAIN_PROCESS_RANK:
            self.file = tempfile.NamedTemporaryFile(delete=True).name
            self.processes = [self._spawn_process(rank) for rank in range(world_size)]
            if crypten.mpc.ttp_required():
                self.processes += [self._spawn_ttp()]

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    @staticmethod
    def _spawn_ttp_process_with_config(config):
        """Runs TTPServer with config copied from parent"""
        cfg.set_config(config)
        crypten.mpc.provider.TTPServer()

    def _spawn_ttp(self):
        communicator_args = {
            "WORLD_SIZE": self.world_size,
            "RANK": self.world_size,
            "RENDEZVOUS": "file://%s" % self.file,
            "BACKEND": "gloo",
        }
        for key, val in communicator_args.items():
            os.environ[key] = str(val)

        process = self.mp_context.Process(
            target=self._spawn_ttp_process_with_config, name="TTP", args=(cfg.config,)
        )
        process.start()
        return process

    def _spawn_process(self, rank):
        name = "Process " + str(rank)
        test_name = self._current_test_name()
        process = self.mp_context.Process(
            target=self.__class__._run,
            name=name,
            args=(test_name, cfg.config, rank, self.world_size, self.file, self.queue),
        )
        process.start()
        return process

    @classmethod
    def _run(cls, test_name, config, rank, world_size, file, exception_queue):
        self = cls(test_name)

        self.file = file
        self.rank = int(rank)
        self.world_size = world_size

        # Copy config to child processes.
        cfg.set_config(config)

        # set environment variables:
        communicator_args = {
            "WORLD_SIZE": self.world_size,
            "RANK": self.rank,
            "RENDEZVOUS": "file://%s" % self.file,
            "BACKEND": "gloo",
        }
        for key, val in communicator_args.items():
            os.environ[key] = str(val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                crypten.init()
            except BaseException:
                tb_string = traceback.format_exc()
                exception_queue.put(tb_string)
                sys.exit(0)
        self.setUp()

        try:
            getattr(self, test_name)()
            exception_queue.put(None)
        except BaseException:
            tb_string = traceback.format_exc()
            exception_queue.put(tb_string)
        crypten.uninit()
        sys.exit(0)

    def _join_processes(self, fn):
        exceptions = {}
        for p in self.processes:
            p.join()
            if not self.queue.empty():
                tb = self.queue.get()
                if tb is not None:
                    exceptions[p.name] = tb

        test_name = str(self.__class__).split("'")[1]
        test_name += f".{self._current_test_name()}"

        msg = f"\n\n\n~ Test {test_name} failed ~"
        msg += "\n===========\nExceptions:\n===========\n"
        for name, tb in exceptions.items():
            msg += f"** {name} ** :\n{tb}\n"

        self.assertEqual(len(exceptions), 0, msg)
