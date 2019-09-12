#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch


@mpc.run_multiprocess(world_size=2)
def test_rank():
    return comm.get().get_rank()


@mpc.run_multiprocess(world_size=2)
def test_exception():
    raise RuntimeError()


@mpc.run_multiprocess(world_size=10)
def test_worldsize():
    return 1


@mpc.run_multiprocess(world_size=2)
def test_generator():
    t0 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g0).item()
    t1 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g1).item()
    return (t0, t1)


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

    def testInitFirst(self):
        # This will cause the parent process to init with world-size 1
        crypten.init()
        self.assertEqual(comm.get().get_world_size(), 1)

        # This will fork 2 children which will have to init with world-size 2
        self.assertEqual(test_rank(), [0, 1])

        # Make sure everything is the same in the parent
        self.assertEqual(comm.get().get_world_size(), 1)

    def testGenerator(self):
        """Tests that generators for PRZS RNG are setup properly with different
        RNG seeds for each process.
        """
        generators = test_generator()

        # Test that generators are communicated properly across processes
        for i in range(len(generators)):
            j = (i + 1) % len(generators)  # Next process index
            self.assertEqual(generators[i][0], generators[j][1])

        # Test that RNG seeds are different from each process
        for i in range(len(generators)):
            self.assertNotEqual(generators[i][0], generators[i][1])
