#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import crypten.communicator as comm
import torch


# TODO: Commenting this out until we figure out why `thread.join() hangs
#       Perhaps the thread to be joined has somehow exited
# from test.multithread_test_case import MultiThreadTestCase


class TestCommunicator:
    """
        This class tests all member functions of crypten package
    """

    def test_przs_generators(self):
        """Tests that przs generators are initialized independently"""
        # Check that each party has two unique generators for g0 and g1
        t0 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g0)
        t1 = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,), generator=comm.get().g1)
        self.assertNotEqual(t0.item(), t1.item())

        # Check that generators are sync'd as expected
        for rank in range(self.world_size):
            receiver = rank
            sender = (rank + 1) % self.world_size
            if self.rank == receiver:
                sender_value = comm.get().recv_obj(sender)
                receiver_value = comm.get().g1.initial_seed()
                self.assertEqual(sender_value, receiver_value)
            elif self.rank == sender:
                sender_value = comm.get().g0.initial_seed()
                comm.get().send_obj(sender_value, receiver)

    def test_global_generator(self):
        """Tests that global generator is generated properly"""
        # Check that all seeds are the same
        this_generator = comm.get().global_generator.initial_seed()
        generator0 = comm.get().broadcast_obj(this_generator, src=0)
        self.assertEqual(this_generator, generator0)

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
                self.assertIsNone(result[0])

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
                    self.assertIsNone(result[0])

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

    def test_batched_all_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        tensors = [get_random_test_tensor(size=size) for size in sizes]

        results = comm.get().all_reduce(tensors, batched=True)
        self.assertTrue(isinstance(results, list))
        for i, result in enumerate(results):
            self.assertTrue((result == (tensors[i] * self.world_size)).all())

    def test_batched_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for rank in range(self.world_size):
            tensors = [get_random_test_tensor(size=size) for size in sizes]
            results = comm.get().reduce(tensors, rank, batched=True)

            if rank == self.rank:
                self.assertTrue(isinstance(results, list))
                for i, result in enumerate(results):
                    self.assertTrue((result == (tensors[i] * self.world_size)).all())
                # NOTE: torch.distributed has undefined behavior for non-dst rank
                # else:
                #     self.assertTrue((result == tensor).all())

    def test_batched_broadcast(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for rank in range(self.world_size):
            if self.rank == rank:
                tensors = [torch.ones(size) for size in sizes]
            else:
                tensors = [torch.zeros(size) for size in sizes]

            tensors = comm.get().broadcast(tensors, src=rank, batched=True)
            self.assertTrue(isinstance(tensors, list))
            for tensor in tensors:
                self.assertTrue(torch.is_tensor(tensor))
                self.assertTrue(tensor.eq(1).all())

    def test_send_recv_obj(self):
        reference = {"a": 1, "b": 2, "c": 3}
        for src in range(self.world_size):
            if self.rank == src:
                test_obj = reference
                comm.get().send_obj(test_obj, 1 - self.rank)
            else:
                test_obj = comm.get().recv_obj(1 - self.rank)
            self.assertEqual(test_obj, reference, "send/recv_obj failed")

    def test_broadcast_obj(self):
        reference = {"a": 1, "b": 2, "c": 3}
        for src in range(self.world_size):
            test_obj = reference if self.rank == src else None
            test_obj = comm.get().broadcast_obj(test_obj, src)
            self.assertEqual(test_obj, reference, "broadcast_obj failed")

    def test_name(self):
        # Test default name is correct
        self.assertEqual(comm.get().get_name(), f"rank{comm.get().get_rank()}")

        # Test name set / get
        comm.get().set_name(f"{comm.get().get_rank()}")
        self.assertEqual(comm.get().get_name(), f"{comm.get().get_rank()}")

        # Test initialization using crypten.init()
        name = f"init_{comm.get().get_rank()}"
        crypten.uninit()
        crypten.init(party_name=name)
        self.assertEqual(comm.get().get_name(), f"init_{comm.get().get_rank()}")

        # Test failure on bad input
        for improper_input in [0, None, ["name"], ("name",)]:
            with self.assertRaises(AssertionError):
                comm.get().set_name(improper_input)


# TODO: Commenting this out until we figure out why `thread.join() hangs
#       Perhaps the thread to be joined has somehow exited
# class TestCommunicatorMultiThread(TestCommunicator, MultiThreadTestCase):
#    pass


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


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
