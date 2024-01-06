#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import crypten


try:
    from ..benchmarks import benchmark
except ValueError:
    # ValueError is raised for relative import
    # when calling $python -m unittest test/test_benchmark.py
    from benchmarks import benchmark


class TestBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        crypten.init()

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_func_benchmarks_run(self) -> None:
        """Ensure function benchmarks run without an exception"""
        func_benchmarks = benchmark.FuncBenchmarks()
        func_benchmarks.run()

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_model_benchmarks_run(self) -> None:
        """Ensure model benchmarks run without an exception"""
        model_benchmarks = benchmark.ModelBenchmarks()
        for model in model_benchmarks.models:
            model.epochs = 2
        model_benchmarks.run()

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_func_benchmarks_data(self) -> None:
        """Sanity check length and columns of function benchmarks"""
        func_benchmarks = benchmark.FuncBenchmarks()
        func_benchmarks.run()
        expected_n_rows = len(benchmark.FuncBenchmarks.UNARY)
        expected_n_rows += len(benchmark.FuncBenchmarks.BINARY)
        expected_n_rows += len(benchmark.FuncBenchmarks.LAYERS)
        n_rows = func_benchmarks.df.shape[0]
        self.assertEqual(
            n_rows,
            expected_n_rows,
            msg=f"function benchmarks {n_rows} rows. Expected {expected_n_rows}",
        )
        self.assertGreater(
            func_benchmarks.df["total abs error"].sum(),
            0,
            msg="total abs error should be greater than 0",
        )
        self.assertTrue(
            all(func_benchmarks.df["runtime"] > 0),
            msg="runtime is less than or equal to zero",
        )
        self.assertTrue(
            all(func_benchmarks.df["runtime crypten"] > 0),
            msg="crypten runtime is less than or equal to zero",
        )

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_model_benchmarks_data(self) -> None:
        """Sanity check length and columns of model benchmarks"""
        model_benchmarks = benchmark.ModelBenchmarks()
        for model in model_benchmarks.models:
            model.epochs = 2
        model_benchmarks.run()
        expected_n_rows = 2 * len(model_benchmarks.models)
        n_rows = model_benchmarks.df.shape[0]
        self.assertEqual(
            n_rows,
            expected_n_rows,
            msg=f"model benchmarks have {n_rows} rows. Expected {expected_n_rows}",
        )
        self.assertTrue(
            all(model_benchmarks.df["seconds per epoch"] > 0),
            msg="seconds per epoch should be greater than 0",
        )
        self.assertTrue(
            all(model_benchmarks.df["inference time"] > 0),
            msg="inference time should be greater than 0",
        )
        self.assertTrue(
            all(model_benchmarks.df["accuracy"] > 0)
            and all(model_benchmarks.df["accuracy"] < 1.0),
            msg="accuracy should be between 0 and 1.0",
        )

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_advanced_model_benchmarks(self) -> None:
        """Tests advanced models are added with flag"""
        model_benchmarks = benchmark.ModelBenchmarks(advanced_models=False)
        self.assertTrue(all(not model.advanced for model in model_benchmarks.models))
        all_model_benchmarks = benchmark.ModelBenchmarks(advanced_models=True)
        self.assertGreater(
            len(all_model_benchmarks.models), len(model_benchmarks.models)
        )


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
