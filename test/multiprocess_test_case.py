#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from functools import wraps
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist


class BenchmarkRun(NamedTuple):
    name: str
    niters: int
    time: float


def get_random_test_tensor(max_value=6, size=(1, 5), is_float=False):
    if is_float:
        tensor = (2 * torch.rand(torch.Size(size)) - 1) * max_value
    else:
        tensor = torch.randint(
            -max_value, max_value, torch.Size(size), dtype=torch.int64
        )

    if dist.is_initialized():
        # Broadcast this tensor to the world so that the generated random tensor
        # is in sync in all distributed processes. See T45688819 for more
        # information.
        dist.broadcast(tensor, 0)

    return tensor


def get_random_linear(in_channels, out_channels):
    linear = torch.nn.Linear(in_channels, out_channels)
    if dist.is_initialized():
        # Broadcast this tensor to the world so that the generated random tensor
        # is in sync in all distributed processes. See T45688819 for more
        # information.
        dist.broadcast(linear.weight, 0)
        dist.broadcast(linear.bias, 0)

    return linear


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK = -1
    _benchmark_results = defaultdict(list)

    @property
    def world_size(self):
        return 2

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
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            logging.warning("Failed to set start method to spawn")
            pass

        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def __init__(self, methodName):
        super().__init__(methodName)

        self.rank = self.MAIN_PROCESS_RANK

    def setUp(self):
        super(MultiProcessTestCase, self).setUp()

        self.benchmark_iters = 100 if self.benchmarks_enabled else 1
        self.default_tolerance = 0.5

        # This gets called in the children process as well to give subclasses a
        # chance to initialize themselves in the new process
        if self.rank == self.MAIN_PROCESS_RANK:
            self.file = tempfile.NamedTemporaryFile(delete=True).name
            self._benchmark_queue = multiprocessing.Queue()

            self.processes = [
                self._spawn_process(rank) for rank in range(int(self.world_size))
            ]

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        self._drain_benchmark_queue()
        for p in self.processes:
            p.terminate()

    @classmethod
    def tearDownClass(cls):
        cls._print_benchmark_summary()

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _spawn_process(self, rank):
        name = "process " + str(rank)
        test_name = self._current_test_name()
        process = multiprocessing.Process(
            target=self.__class__._run,
            name=name,
            args=(test_name, rank, self._benchmark_queue, self.file),
        )
        process.start()
        return process

    @classmethod
    def _run(cls, test_name, rank, queue, file):
        self = cls(test_name)

        self.file = file
        self.rank = int(rank)
        self._benchmark_queue = queue

        # set environment variables:
        communicator_args = {
            "WORLD_SIZE": self.world_size,
            "RANK": self.rank,
            "RENDEZVOUS": "file://%s" % self.file,
            "BACKEND": "gloo",
        }
        for key, val in communicator_args.items():
            os.environ[key] = str(val)

        self.setUp()

        # We're retrieving a corresponding test and executing it.
        getattr(self, test_name)()
        sys.exit(0)

    def _join_processes(self, fn):
        for p in self.processes:
            p.join()
            self._check_return_codes(p)

    def _check_return_codes(self, process):
        self.assertEqual(process.exitcode, 0)

    def _add_benchmark_results(self, rank, args, time, niters):
        args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        benchmark_name = f"{self._current_test_name()} ({args_str})"
        self._benchmark_queue.put(
            BenchmarkRun(name=benchmark_name, niters=niters, time=time)
        )

    def benchmark(self, niters=None, data=None, **kwargs):
        class Benchmark:
            def __init__(self, test_case, niters, data=None, **kwargs):
                if data is not None:
                    niters = len(data)
                    self.data = data

                self.niters = niters
                self.iters = range(self.niters)
                self.args = kwargs
                self.test_case = test_case

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, etype, evalue, etraceback):
                self.end_time = time.perf_counter()
                self.test_case._add_benchmark_results(
                    self.test_case.rank,
                    self.args,
                    self.end_time - self.start_time,
                    self.niters,
                )

        if niters is None:
            niters = self.benchmark_iters
        if not self.benchmarks_enabled:
            niters = 1

        return Benchmark(data=data, niters=niters, test_case=self, **kwargs)

    def _drain_benchmark_queue(self):
        while not self._benchmark_queue.empty():
            run = self._benchmark_queue.get()
            MultiProcessTestCase._benchmark_results[run.name].append(run)

    @classmethod
    def _print_benchmark_summary(cls):
        if not cls.benchmarks_enabled:
            return

        def format_time(time: float) -> str:
            units = ["s", "ms", "us", "ns"]
            idx = 0
            while time < 1 and idx < len(units):
                time *= 1000
                idx += 1
            return f"{time:10.3f}{units[idx]}"

        def log(message: str):
            print(message, file=sys.stderr)

        log(f"Benchmark summary for {cls.__name__}")
        log("-" * 80)

        name = "Benchmark"
        time = "Time per iteration"
        niters = "Iterations"
        iters_sec = "Iterations per second"

        log(f"{name:<100}{time:>20}{niters:>20}{iters_sec:>25}")
        log("-" * 80)
        for name, runs in MultiProcessTestCase._benchmark_results.items():
            avg_time = np.mean([x.time for x in runs])
            niters = runs[0].niters
            time = format_time(avg_time / niters)
            iters_sec = int(niters / avg_time)

            log(f"{name:<100}{time:>20}{niters:>20}{iters_sec:>25}")
        log("-" * 80)
