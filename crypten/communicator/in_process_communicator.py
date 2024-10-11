#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import threading
from operator import itemgetter
from queue import Queue

import torch
from torch.distributed import ReduceOp

from .communicator import Communicator


class InProcessCommunicator(Communicator):
    BYTES_PER_ELEMENT = 8
    tls = threading.local()
    mailbox = None
    barrier = None
    lock = threading.Lock()

    @classmethod
    def initialize(cls, rank, world_size, init_ttp=False):
        cls.tls.instance = cls(rank, world_size)

    def __init__(self, rank, world_size, init_ttp=False):
        self.world_size = world_size
        self.rank = rank
        self.reset_communication_stats()
        self._name = f"rank{rank}"

        with InProcessCommunicator.lock:
            if InProcessCommunicator.mailbox is None:
                InProcessCommunicator.mailbox = [
                    Queue() for _ in range(self.world_size)
                ]

                # This prevents one thread from running ahead of the others and doing
                # multiple puts that would show up in the get calls below
                InProcessCommunicator.barrier = threading.Barrier(self.world_size)

        # logging:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        logging.info("==================")
        logging.info("InProcessCommunicator with rank %d" % self.rank)
        logging.info("==================")

        logging.info("World size = %d" % self.get_world_size())
        logging.getLogger().setLevel(level)

    @classmethod
    def get(cls):
        if not hasattr(cls.tls, "instance"):
            return None

        return cls.tls.instance

    @classmethod
    def is_initialized(cls):
        return hasattr(cls.tls, "instance")

    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        self.mailbox[dst].put((self.rank, tensor.clone()))

    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        rank, result = self.mailbox[self.rank].get()
        if src is not None and rank != src:
            raise NotImplementedError("Can't receive messages out of order yet")
        return result

    def isend(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        self.send(tensor, dst)

        class Result:
            def is_completed(self):
                return True

            def wait(self):
                pass

        return Result()

    def irecv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""

        class Result:
            def __init__(self, mailbox, rank):
                self.completed = False
                self.mailbox = mailbox
                self.rank = rank

            def is_completed(self):
                return self.completed

            def wait(self):
                rank, result = self.mailbox[self.rank].get()
                if src is not None and rank != src:
                    raise NotImplementedError("Can't receive messages out of order yet")
                tensor.copy_(result)

        return Result(self.mailbox, self.rank)

    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of tensors to all parties."""
        if async_op:
            raise NotImplementedError()

        if src == self.rank:
            for i in range(self.world_size):
                self.mailbox[i].put(scatter_list[i].clone())

        self.barrier.wait()

        return self.mailbox[self.rank].get()

    def reduce(self, tensor, dst, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties."""
        tensors = self.gather(tensor, dst)
        if self.rank == dst:
            reduce_fn = self._reduce_op_to_function(op)
            return reduce_fn(torch.stack(tensors), dim=0)

    @classmethod
    def shutdown(cls):
        # Destroy all thread-local instances
        cls.tls = threading.local()
        cls.mailbox = None
        cls.barrier = None

    def _reduce_op_to_function(self, op):
        if op == ReduceOp.SUM:
            return torch.sum

        raise NotImplementedError()

    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        if async_op:
            raise NotImplementedError()

        ag = self.all_gather(tensor)
        reduce_fn = self._reduce_op_to_function(op)
        return reduce_fn(torch.stack(ag), dim=0)

    def gather(self, tensor, dst, async_op=False):
        """Gathers a list of tensors in a single party."""
        if async_op:
            raise NotImplementedError()

        self.mailbox[dst].put((self.rank, tensor.clone()))

        self.barrier.wait()

        if self.rank == dst:
            result = [self.mailbox[dst].get() for _ in range(self.world_size)]
            return [tensor for rank, tensor in sorted(result, key=itemgetter(0))]

    def all_gather(self, tensor, async_op=False):
        """Gathers tensors from all parties in a list."""
        if async_op:
            raise NotImplementedError()

        for i in range(self.world_size):
            self.mailbox[i].put((self.rank, tensor.clone()))

        self.barrier.wait()

        result = sorted(
            (self.mailbox[self.rank].get() for _ in range(self.world_size)),
            key=itemgetter(0),
        )

        return [tensor for (rank, tensor) in result]

    def broadcast(self, tensor, src, async_op=False):
        """Broadcasts the tensor to all parties."""
        if async_op:
            raise NotImplementedError()

        if self.rank == src:
            for i in range(self.get_world_size()):
                self.mailbox[i].put(tensor.clone())

        # No need for a barrier here.

        return self.mailbox[self.rank].get()

    def get_world_size(self):
        """Returns the size of the world."""
        return self.world_size

    def get_rank(self):
        """Returns the rank of the current process."""
        return self.rank

    def set_name(self, name):
        """Sets the party name of the current rank."""
        assert isinstance(
            name, str
        ), f"Improper name provided to process on rank {self.get_rank()}"
        self._name = name

    def get_name(self):
        """Returns the party name of the current rank."""
        return self._name
