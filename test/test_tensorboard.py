#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.multiprocess_test_case import MultiProcessTestCase

import crypten
import crypten.nn.tensorboard as tensorboard


class TestTensorboard(MultiProcessTestCase):
    """This class tests the crypten.nn.tensorboad package."""

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        if self.rank >= 0:
            crypten.init()

    def test_tensorboard(self):

        # create small crypten model:
        model = crypten.nn.Graph("input", "output")
        model.add_module("intermediate1", crypten.nn.ReLU(), ["input"])
        model.add_module("intermediate2", crypten.nn.Constant(1), [])
        model.add_module("output", crypten.nn.Add(), ["intermediate1", "intermediate2"])

        # create tensorboard graph:
        tensorboard.graph(model)
        self.assertTrue(True, "creation of tensorboard graph failed")


if __name__ == "__main__":
    unittest.main()
