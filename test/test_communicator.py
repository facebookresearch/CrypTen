#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import crypten
import crypten.communicator as comm
import numpy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from crypten.common import serial
from crypten.config import cfg
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


# TODO: Commenting this out until we figure out why `thread.join() hangs
#       Perhaps the thread to be joined has somehow exited
# from test.multithread_test_case import MultiThreadTestCase

INVALID_SERIALIZED_OBJECTS = [
    b"cos\nsystem\n(S'echo hello world'\ntR.",
    b'\x80\x03cbuiltins\neval\n(Vprint("I should not print")\ntRctorch._utils\n_rebuild_tensor_v2\n(ctorch.storage\n_load_from_bytes\nB\x01\x01\x00\x00\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.\x80\x02}q\x00(X\x10\x00\x00\x00protocol_versionq\x01M\xe9\x03X\r\x00\x00\x00little_endianq\x02\x88X\n\x00\x00\x00type_sizesq\x03}q\x04(X\x05\x00\x00\x00shortq\x05K\x02X\x03\x00\x00\x00intq\x06K\x04X\x04\x00\x00\x00longq\x07K\x04uu.\x80\x02(X\x07\x00\x00\x00storageq\x00ctorch\nFloatStorage\nq\x01X\x0f\x00\x00\x00140436995850160q\x02X\x03\x00\x00\x00cpuq\x03K\x01Ntq\x04Q.\x80\x02]q\x00X\x0f\x00\x00\x00140436995850160q\x01a.\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x85RK\x00K\x01\x85K\x01\x85\x89ccollections\nOrderedDict\n)RtR.',
    b'\x80\x04\x8c\x12torch.__builtins__\x94\x8c\x03get\x94\x93\x8c\x04eval\x94\x85\x94R\x8c\x1bprint("I should not print")\x94\x85\x94R.',
    b"\x80\x04\x8c$torch.nn.modules._functions.torch.os\x94\x8c\x05execl\x94\x93\x8c\x0c/usr/bin/vim\x94\x8c\x0c/usr/bin/vim\x94\x86\x94R.",
    b"\x80\x04\x8c\rtorch.storage\x94\x8c\x10_load_from_bytes\x94\x93C2\x80\x04\x8c\x02os\x94\x8c\x05execl\x94\x93\x8c\x0c/usr/bin/vim\x94\x8c\x0c/usr/bin/vim\x94\x86\x94R.\x94\x85\x94R.",
]


class TestCommunicator:
    """
    This class tests all member functions of crypten package
    """

    def test_send_recv(self):
        tensor = torch.tensor([self.rank], dtype=torch.long)

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
            tensor = torch.tensor([0], dtype=torch.long)
            if self.rank == rank:
                tensor += 1

            tensor = comm.get().broadcast(tensor, rank)
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

            tensors = comm.get().broadcast(tensors, rank, batched=True)
            self.assertTrue(isinstance(tensors, list))
            for tensor in tensors:
                self.assertTrue(torch.is_tensor(tensor))
                self.assertTrue(tensor.eq(1).all())

    def test_send_recv_obj(self):
        TEST_OBJECTS = [
            {"a": 1, "b": 2, "c": 3},
            torch.tensor(1),
            torch.nn.Linear(10, 5),
            CNN(),
        ]
        for param in TEST_OBJECTS[2].parameters():
            param.data.fill_(1.0)
        for param in TEST_OBJECTS[3].parameters():
            param.data.fill_(1.0)
        serial.register_safe_class(CNN)

        for reference in TEST_OBJECTS:
            for src in range(self.world_size):
                if self.rank == src:
                    test_obj = reference
                    comm.get().send_obj(test_obj, 1 - self.rank)
                else:
                    test_obj = comm.get().recv_obj(1 - self.rank)

                if isinstance(reference, torch.nn.Module):
                    test_obj_params = list(test_obj.parameters())
                    reference_params = list(reference.parameters())
                    for i, param in enumerate(reference_params):
                        self.assertTrue(
                            test_obj_params[i].eq(param).all(), "broadcast_obj failed"
                        )
                else:
                    self.assertEqual(test_obj, reference, "broadcast_obj failed")

        # Test that the restricted loader will raise an error for code injection
        for invalid_obj in INVALID_SERIALIZED_OBJECTS:
            for src in range(self.world_size):
                if self.rank == src:
                    # Mimic send_obj without pickling invalid bytestream
                    size = torch.tensor(len(invalid_obj), dtype=torch.int32)
                    arr = torch.from_numpy(
                        numpy.frombuffer(invalid_obj, dtype=numpy.int8)
                    )

                    r0 = dist.isend(
                        size, dst=(1 - self.rank), group=comm.get().main_group
                    )
                    r1 = dist.isend(
                        arr, dst=(1 - self.rank), group=comm.get().main_group
                    )

                    r0.wait()
                    r1.wait()
                else:
                    with self.assertRaises(ValueError):
                        comm.get().recv_obj(1 - self.rank)

    def test_broadcast_obj(self):
        TEST_OBJECTS = [
            {"a": 1, "b": 2, "c": 3},
            torch.tensor(1),
            torch.nn.Linear(10, 5),
            CNN(),
        ]
        for param in TEST_OBJECTS[2].parameters():
            param.data.fill_(1.0)
        for param in TEST_OBJECTS[3].parameters():
            param.data.fill_(1.0)
        serial.register_safe_class(CNN)

        for reference in TEST_OBJECTS:
            for src in range(self.world_size):
                test_obj = reference if self.rank == src else None
                test_obj = comm.get().broadcast_obj(test_obj, src)
                if isinstance(reference, torch.nn.Module):
                    test_obj_params = list(test_obj.parameters())
                    reference_params = list(reference.parameters())
                    for i, param in enumerate(reference_params):
                        self.assertTrue(
                            test_obj_params[i].eq(param).all(), "broadcast_obj failed"
                        )
                else:
                    self.assertEqual(test_obj, reference, "broadcast_obj failed")

        # Test that the restricted loader will raise an error for code injection
        for invalid_obj in INVALID_SERIALIZED_OBJECTS:
            for src in range(self.world_size):
                if self.rank == src:
                    # Mimic broadcast_obj without pickling invalid bytestream
                    size = torch.tensor(len(invalid_obj), dtype=torch.int32)
                    arr = torch.from_numpy(
                        numpy.frombuffer(invalid_obj, dtype=numpy.int8)
                    )

                    dist.broadcast(size, src, group=comm.get().main_group)
                    dist.broadcast(arr, src, group=comm.get().main_group)
                else:
                    with self.assertRaises(ValueError):
                        test_obj = None
                        comm.get().broadcast_obj(test_obj, src)

    @unittest.skip("Skipping for now as it keeps timing out")  # FIXME
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
        cfg.communicator.verbose = True
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
                nbytes = tensor.numel() * 8
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
                if op in ["all_reduce", "all_gather"]:
                    reference = 2 * nbytes * (self.world_size - 1)
                else:
                    reference = nbytes * (self.world_size - 1)
                self.assertEqual(comm.get().comm_bytes, reference)

        # Test reset_communication_stats
        crypten.reset_communication_stats()
        self.assertEqual(comm.get().comm_rounds, 0)
        self.assertEqual(comm.get().comm_bytes, 0)

        # test retrieving communication stats:
        stats = comm.get().get_communication_stats()
        self.assertIsInstance(stats, dict)
        for key in ["rounds", "bytes", "time"]:
            self.assertIn(key, stats)
            self.assertEqual(stats[key], 0)

        # Test verbosity False setting and no logging
        cfg.communicator.verbose = False
        tensor = get_random_test_tensor(size=size, is_float=False)
        tensor = comm.get().broadcast(tensor, 0)
        self.assertEqual(comm.get().comm_rounds, 0)
        self.assertEqual(comm.get().comm_bytes, 0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(16 * 13 * 13, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
