#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor
from test.multithread_test_case import MultiThreadTestCase

import crypten
import crypten.communicator as comm
import torch


class TestCommunicator:
    """
        This class tests all member functions of crypten package
    """

    benchmarks_enabled = False

    def test_przs_generators(self):
        """Tests that przs generators are initialized independently"""
        t0 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g0)
        t1 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g1)
        self.assertNotEqual(t0.item(), t1.item())

    def test_send_recv(self):
        tensor = torch.LongTensor([self.rank])

        # Send forward, receive backward
        dst = (self.rank + 1) % self.world_size
        src = (self.rank - 1) % self.world_size

        if self.rank == 0:
            comm.get().send(tensor, dst=dst)
        result = comm.get().recv(tensor, src=src)
        if self.rank > 0:
            comm.get().send(tensor, dst=dst)

        self.assertTrue(torch.is_tensor(result))
        self.assertEqual(result.item(), src)

    def test_scatter(self):
        for rank in range(self.world_size):
            tensor = []
            if self.rank == rank:
                tensor = [torch.tensor(i) for i in range(self.world_size)]

            result = comm.get().scatter(tensor, rank, size=())
            self.assertTrue(torch.is_tensor(result))
            self.assertEqual(result.item(), self.rank)

    def test_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for rank in range(self.world_size):
            for size in sizes:
                tensor = get_random_test_tensor(size=size)
                result = comm.get().reduce(tensor, rank)

                if rank == self.rank:
                    self.assertTrue((result == (tensor * self.world_size)).all())
                # NOTE: torch.distributed has undefined behavior for non-dst rank
                # else:
                #     self.assertTrue((result == tensor).all())

    def test_all_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size)
            result = comm.get().all_reduce(tensor)
            self.assertTrue((result == (tensor * self.world_size)).all())

    def test_gather(self):
        tensor = torch.tensor([self.rank])
        for rank in range(self.world_size):
            result = comm.get().gather(tensor, rank)
            if rank == self.rank:
                self.assertEqual(result, [torch.tensor([0]), torch.tensor([1])])
            else:
                self.assertIsNone(result)

    def test_gather_random(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5), (1000,)]
        for rank in range(self.world_size):
            for size in sizes:
                tensor = get_random_test_tensor(size=size)
                result = comm.get().gather(tensor, rank)
                if rank == self.rank:
                    self.assertTrue(isinstance(result, list))
                    for res in result:
                        self.assertTrue((res == tensor).all())
                else:
                    self.assertIsNone(result)

    def test_all_gather(self):
        tensor = torch.tensor([self.rank])
        result = comm.get().all_gather(tensor)
        self.assertEqual(
            result, [torch.tensor([rank]) for rank in range(self.world_size)]
        )

    def test_mutation(self):
        for _ in range(10):
            tensor = torch.tensor([self.rank])
            result = comm.get().all_gather(tensor)
            # Mutate the tensor, which should have no effect since the gather
            # has finished. If we don't clone the tensor though, this might
            # mutate one of the tensors received by the other party.
            tensor += 1
            self.assertEqual(
                result, [torch.tensor([rank]) for rank in range(self.world_size)]
            )

    def test_all_gather_random(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size)
            result = comm.get().all_gather(tensor)
            self.assertTrue(isinstance(result, list))
            for res in result:
                self.assertTrue((res == tensor).all())

    def test_broadcast(self):
        for rank in range(self.world_size):
            tensor = torch.LongTensor([0])
            if self.rank == rank:
                tensor += 1

            tensor = comm.get().broadcast(tensor, src=rank)
            self.assertTrue(torch.is_tensor(tensor))
            self.assertEqual(tensor.item(), 1)

    def test_get_world_size(self):
        self.assertEqual(comm.get().get_world_size(), self.world_size)

    def test_get_rank(self):
        self.assertEqual(comm.get().get_rank(), self.rank)


class TestCommunicatorMultiThread(TestCommunicator, MultiThreadTestCase):
    pass


class TestCommunicatorMultiProcess(TestCommunicator, MultiProcessTestCase):
    def test_logging(self):
        # Assert initialization resets comm.get() stats
        self.assertEqual(comm.get().comm_rounds, 0)
        self.assertEqual(comm.get().comm_bytes, 0)

        # Test verbosity True setting and logging
        comm.get().set_verbosity(True)
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]

        # Test send / recv:
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=False)
            crypten.reset_communication_stats()

            # Send forward, receive backward
            dst = (self.rank + 1) % self.world_size
            src = (self.rank - 1) % self.world_size

            if self.rank == 0:
                comm.get().send(tensor, dst=dst)
            tensor = comm.get().recv(tensor, src=src)
            if self.rank > 0:
                comm.get().send(tensor, dst=dst)

            self.assertEqual(comm.get().comm_rounds, 2)
            self.assertEqual(comm.get().comm_bytes, tensor.numel() * 8 * 2)

        # Test all other ops:
        ops = ["all_reduce", "all_gather", "broadcast", "gather", "reduce", "scatter"]
        for size in sizes:
            for op in ops:
                tensor = get_random_test_tensor(size=size, is_float=False)
                bytes = tensor.numel() * 8
                crypten.reset_communication_stats()

                # Setup op-specific kwargs / inputs
                args = ()
                if op in ["gather", "reduce"]:
                    args = (0,)  # dst arg
                if op == "broadcast":
                    args = (0,)  # dst arg
                if op == "scatter":
                    tensor = [tensor] * self.world_size
                    args = (0,)  # src arg

                tensor = getattr(comm.get(), op)(tensor, *args)
                self.assertEqual(comm.get().comm_rounds, 1)
                self.assertEqual(comm.get().comm_bytes, bytes * (self.world_size - 1))

        # Test reset_communication_stats
        crypten.reset_communication_stats()
        self.assertEqual(comm.get().comm_rounds, 0)
        self.assertEqual(comm.get().comm_bytes, 0)

        # Test verbosity False setting and no logging
        comm.get().set_verbosity(False)
        tensor = get_random_test_tensor(size=size, is_float=False)
        tensor = comm.get().broadcast(tensor, src=0)
        self.assertEqual(comm.get().comm_rounds, 0)
        self.assertEqual(comm.get().comm_bytes, 0)


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestCommunicator.benchmarks_enabled = True
    unittest.main()
