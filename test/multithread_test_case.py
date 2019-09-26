#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import queue
import sys
import threading
import unittest
from functools import wraps
from threading import Thread

import crypten

from .benchmark_helper import BenchmarkHelper


class MultiThreadTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK = -1

    @property
    def rank(self):
        from crypten.communicator import InProcessCommunicator

        if threading.current_thread() == threading.main_thread():
            return self.MAIN_PROCESS_RANK

        return InProcessCommunicator.get().rank

    @property
    def world_size(self):
        return 2

    def __init__(self, methodName):
        super().__init__(methodName)
        # TODO(vini): support benchmarking in threaded mode
        self.benchmark_iters = 1
        self.benchmark_enabled = False
        self.default_tolerance = 0.5
        q = queue.Queue()
        self.benchmark_helper = BenchmarkHelper(False, self.world_size, q)

    def benchmark(self, niters=None, data=None, **kwargs):
        return self.benchmark_helper.benchmark(self, niters, data, **kwargs)

    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if threading.current_thread() == threading.main_thread():
                self._join_threads()
            else:
                fn(self)

        return wrapper

    def _join_threads(self):
        for t in self.threads:
            t.join()

        try:
            exception_info = self.exception_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            sys.excepthook(*exception_info)
            raise RuntimeError(
                "Exception found in one of the parties. Look at past logs."
            )

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def setUp(self):
        super().setUp()

        if threading.current_thread() != threading.main_thread():
            return
        test_name = self._current_test_name()
        test_fn = getattr(self, test_name)
        self.exception_queue = queue.Queue()
        self.threads = [
            Thread(target=self._run, args=(test_fn, rank, self.world_size))
            for rank in range(self.world_size)
        ]
        for t in self.threads:
            t.start()

    def _run(self, test_fn, rank, world_size):
        crypten.init_thread(rank, world_size)

        self.setUp()

        try:
            test_fn()
        except Exception:
            self.exception_queue.put(sys.exc_info())
