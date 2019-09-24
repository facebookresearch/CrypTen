#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np


class BenchmarkRun(NamedTuple):
    name: str
    niters: int
    time: float


class BenchmarkHelper:
    _benchmark_results = defaultdict(list)

    def __init__(self, benchmarks_enabled, benchmark_iters, queue):
        self.benchmarks_enabled = benchmarks_enabled
        self.benchmark_iters = benchmark_iters
        self.queue = queue

    def _add_benchmark_results(self, test_name, rank, args, time, niters):
        args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        benchmark_name = f"{test_name} ({args_str})"
        self.queue.put(BenchmarkRun(name=benchmark_name, niters=niters, time=time))

    def benchmark(self, test_case, niters=None, data=None, **kwargs):
        class Benchmark:
            def __init__(self, test_case, helper, niters, data=None, **kwargs):
                if data is not None:
                    niters = len(data)
                    self.data = data

                self.niters = niters
                self.iters = range(self.niters)
                self.args = kwargs
                self.test_case = test_case
                self.helper = helper

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, etype, evalue, etraceback):
                self.end_time = time.perf_counter()
                self.helper._add_benchmark_results(
                    self.test_case._current_test_name(),
                    self.test_case.rank,
                    self.args,
                    self.end_time - self.start_time,
                    self.niters,
                )

        if niters is None:
            niters = test_case.benchmark_iters
        if not test_case.benchmarks_enabled:
            niters = 1

        return Benchmark(
            data=data, niters=niters, test_case=test_case, helper=self, **kwargs
        )

    def drain_benchmark_queue(self):
        while not self.queue.empty():
            run = self.queue.get()
            self.__class__._benchmark_results[run.name].append(run)

    def print_benchmark_summary(self, name):
        if not self.benchmarks_enabled:
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

        log(f"Benchmark summary for {name}")
        log("-" * 80)

        name = "Benchmark"
        time = "Time per iteration"
        niters = "Iterations"
        iters_sec = "Iterations per second"

        log(f"{name:<100}{time:>20}{niters:>20}{iters_sec:>25}")
        log("-" * 80)
        for name, runs in self.__class__._benchmark_results.items():
            avg_time = np.mean([x.time for x in runs])
            niters = runs[0].niters
            time = format_time(avg_time / niters)
            iters_sec = int(niters / avg_time)

            log(f"{name:<100}{time:>20}{niters:>20}{iters_sec:>25}")
        log("-" * 80)
