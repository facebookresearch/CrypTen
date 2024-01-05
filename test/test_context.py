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
from crypten.config import cfg


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
    device = torch.device("cpu")
    t0 = torch.randint(
        -(2**63), 2**63 - 1, (1,), generator=crypten.generators["prev"][device]
    ).item()
    t1 = torch.randint(
        -(2**63), 2**63 - 1, (1,), generator=crypten.generators["next"][device]
    ).item()
    return (t0, t1)


@mpc.run_multiprocess(world_size=2)
def test_with_args_kwargs_func(first, *args, a=None, **kwargs):
    """function that removes first arg and `a` kwarg"""
    return args, kwargs


@mpc.run_multiprocess(world_size=5)
def test_rng_seeds_func():
    """Tests that rng seeds differ and coordinate where desired"""
    device = torch.device("cpu")
    prev_seed = crypten.generators["prev"][device].initial_seed()
    next_seed = crypten.generators["next"][device].initial_seed()
    local_seed = crypten.generators["local"][device].initial_seed()
    global_seed = crypten.generators["global"][device].initial_seed()

    return (prev_seed, next_seed, local_seed, global_seed)


class TestContext(unittest.TestCase):
    def test_rank(self) -> None:
        ranks = test_rank_func()
        self.assertEqual(ranks, [0, 1])

    def test_exception(self) -> None:
        ret = test_exception_func()
        self.assertEqual(ret, None)

    def test_world_size(self) -> None:
        ones = test_worldsize_func()
        self.assertEqual(ones, [1] * 10)

    def test_in_first(self) -> None:
        # TODO: Make this work with TTP provider
        cfg.mpc.provider = "TFP"

        # This will cause the parent process to init with world-size 1
        crypten.init()
        self.assertEqual(comm.get().get_world_size(), 1)

        # This will fork 2 children which will have to init with world-size 2
        self.assertEqual(test_rank_func(), [0, 1])

        # Make sure everything is the same in the parent
        self.assertEqual(comm.get().get_world_size(), 1)

    def test_generator(self) -> None:
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

    def test_with_args_kwargs(self) -> None:
        args = (2, 3, 5, 8)
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        retval = test_with_args_kwargs_func(*args, **kwargs)

        ret_args, ret_kwargs = retval[0]
        kwargs.pop("a")

        self.assertEqual(ret_args, args[1:])
        self.assertEqual(ret_kwargs, kwargs)

    def test_rng_seeds(self) -> None:
        all_seeds = test_rng_seeds_func()

        prev_seeds = [seed[0] for seed in all_seeds]
        next_seeds = [seed[1] for seed in all_seeds]
        local_seeds = [seed[2] for seed in all_seeds]
        global_seeds = [seed[3] for seed in all_seeds]

        # Test local seeds are all unique
        self.assertTrue(len(set(local_seeds)) == len(local_seeds))

        # Test global seeds are all the same
        self.assertTrue(len(set(global_seeds)) == 1)

        # Test that next seeds are equal to next party's prev_seed
        for i, next_seed in enumerate(next_seeds):
            next_index = (i + 1) % len(prev_seeds)
            prev_seed = prev_seeds[next_index]

            self.assertEqual(next_seed, prev_seed)
