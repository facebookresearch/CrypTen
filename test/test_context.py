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
def test_rank_func():
    return comm.get().get_rank()


@mpc.run_multiprocess(world_size=2)
def test_exception_func():
    raise RuntimeError()


@mpc.run_multiprocess(world_size=10)
def test_worldsize_func():
    return 1


@mpc.run_multiprocess(world_size=2)
def test_generator_func():
    t0 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g0).item()
    t1 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g1).item()
    return (t0, t1)


@mpc.run_multiprocess(world_size=2)
def test_with_args_kwargs_func(first, *args, a=None, **kwargs):
    """function that removes first arg and `a` kwarg"""
    return args, kwargs


class TestContext(unittest.TestCase):
    def test_rank(self):
        ranks = test_rank_func()
        self.assertEqual(ranks, [0, 1])

    def test_exception(self):
        ret = test_exception_func()
        self.assertEqual(ret, None)

    def test_world_size(self):
        ones = test_worldsize_func()
        self.assertEqual(ones, [1] * 10)

    def test_in_first(self):
        # This will cause the parent process to init with world-size 1
        crypten.init()
        self.assertEqual(comm.get().get_world_size(), 1)

        # This will fork 2 children which will have to init with world-size 2
        self.assertEqual(test_rank_func(), [0, 1])

        # Make sure everything is the same in the parent
        self.assertEqual(comm.get().get_world_size(), 1)

    def test_generator(self):
        """Tests that generators for PRZS RNG are setup properly with different
        RNG seeds for each process.
        """
        generators = test_generator_func()

        # Test that generators are communicated properly across processes
        for i in range(len(generators)):
            j = (i + 1) % len(generators)  # Next process index
            self.assertEqual(generators[i][0], generators[j][1])

        # Test that RNG seeds are different from each process
        for i in range(len(generators)):
            self.assertNotEqual(generators[i][0], generators[i][1])

    def test_with_args_kwargs(self):
        args = (2, 3, 5, 8)
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        retval = test_with_args_kwargs_func(*args, **kwargs)

        ret_args, ret_kwargs = retval[0]
        kwargs.pop("a")

        self.assertEqual(ret_args, args[1:])
        self.assertEqual(ret_kwargs, kwargs)
