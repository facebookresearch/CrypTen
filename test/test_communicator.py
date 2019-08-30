#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor


# placeholders for class references, to be filled later by import_crypten():
crypten, comm = None, None


def import_crypten():
    """
    Imports CrypTen types. This function is called after environment variables
    in MultiProcessTestCase.setUp() are set, and sets the class references for
    all test functions.
    """
    global crypten, comm
    import crypten as _crypten
    from crypten import comm as _comm

    crypten = _crypten
    comm = _comm


class TestCommunicator(MultiProcessTestCase):
    """
        This class tests all member functions of crypten package
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        if self.rank >= 0:
            import_crypten()

    def test_send_recv(self):
        tensor = torch.LongTensor([self.rank])

        # Send forward, receive backward
        dst = (self.rank + 1) % self.world_size
        src = (self.rank - 1) % self.world_size

        if self.rank == 0:
            comm.send(tensor, dst=dst)
        result = comm.recv(tensor, src=src)
        if self.rank > 0:
            comm.send(tensor, dst=dst)

        self.assertTrue(torch.is_tensor(result))
        self.assertEqual(result.item(), src)

    def test_scatter(self):
        for rank in range(self.world_size):
            tensor = []
            if self.rank == rank:
                tensor = [torch.tensor(i) for i in range(self.world_size)]

            result = comm.scatter(tensor, rank, size=())
            self.assertTrue(torch.is_tensor(result))
            self.assertEqual(result.item(), self.rank)

    def test_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for rank in range(self.world_size):
            for size in sizes:
                tensor = get_random_test_tensor(size=size)
                result = comm.reduce(tensor, rank)

                if rank == self.rank:
                    self.assertTrue((result == (tensor * self.world_size)).all())
                # NOTE: torch.distributed has undefined behavior for non-dst rank
                # else:
                #     self.assertTrue((result == tensor).all())

    def test_all_reduce(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size)
            result = comm.all_reduce(tensor)
            self.assertTrue((result == (tensor * self.world_size)).all())

    def test_gather(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for rank in range(self.world_size):
            for size in sizes:
                tensor = get_random_test_tensor(size=size)
                result = comm.gather(tensor, rank)
                if rank == self.rank:
                    self.assertTrue(isinstance(result, list))
                    for res in result:
                        self.assertTrue((res == tensor).all())
                else:
                    self.assertIsNone(result)

    def test_all_gather(self):
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size)
            result = comm.all_gather(tensor)
            self.assertTrue(isinstance(result, list))
            for res in result:
                self.assertTrue((res == tensor).all())

    def test_broadcast(self):
        for rank in range(self.world_size):
            tensor = torch.LongTensor([0])
            if self.rank == rank:
                tensor += 1

            tensor = comm.broadcast(tensor, src=rank)
            self.assertTrue(torch.is_tensor(tensor))
            self.assertEqual(tensor.item(), 1)

    def test_get_world_size(self):
        self.assertEqual(comm.get_world_size(), self.world_size)

    def test_get_rank(self):
        self.assertEqual(comm.get_rank(), self.rank)

    def test_logging(self):
        # Assert initialization resets communication stats
        self.assertEqual(comm.comm_rounds, 0)
        self.assertEqual(comm.comm_bytes, 0)

        # Test verbosity True setting and logging
        comm.set_verbosity(True)
        sizes = [(), (1,), (5,), (5, 5), (5, 5, 5)]

        # Test send / recv:
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=False)
            crypten.reset_communication_stats()

            # Send forward, receive backward
            dst = (self.rank + 1) % self.world_size
            src = (self.rank - 1) % self.world_size

            if self.rank == 0:
                comm.send(tensor, dst=dst)
            tensor = comm.recv(tensor, src=src)
            if self.rank > 0:
                comm.send(tensor, dst=dst)

            self.assertEqual(comm.comm_rounds, 2)
            self.assertEqual(comm.comm_bytes, tensor.numel() * 8 * 2)

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
                    args = (0, )        # dst arg
                if op == "broadcast":
                    args = (0, )        # dst arg
                if op == "scatter":
                    tensor = [tensor] * self.world_size
                    args = (0, )        # src arg

                tensor = getattr(comm, op)(tensor, *args)
                self.assertEqual(comm.comm_rounds, 1)
                self.assertEqual(comm.comm_bytes, bytes * (self.world_size - 1))

        # Test reset_communication_stats
        crypten.reset_communication_stats()
        self.assertEqual(comm.comm_rounds, 0)
        self.assertEqual(comm.comm_bytes, 0)

        # Test verbosity False setting and no logging
        comm.set_verbosity(False)
        tensor = get_random_test_tensor(size=size, is_float=False)
        tensor = comm.broadcast(tensor, src=0)
        self.assertEqual(comm.comm_rounds, 0)
        self.assertEqual(comm.comm_bytes, 0)


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestCommunicator.benchmarks_enabled = True
    unittest.main()
