#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import crypten.communicator as comm
import crypten.mpc as mpc


@mpc.run_multiprocess(world_size=2)
def test_rank():
    return comm.get().get_rank()


@mpc.run_multiprocess(world_size=2)
def test_exception():
    raise RuntimeError()


@mpc.run_multiprocess(world_size=10)
def test_worldsize():
    return 1


class ContextTest(unittest.TestCase):
    def testRank(self):
        ranks = test_rank()
        self.assertEqual(ranks, [0, 1])

    def testException(self):
        ret = test_exception()
        self.assertEqual(ret, None)

    def testWorldSize(self):
        ones = test_worldsize()
        self.assertEqual(ones, [1] * 10)
