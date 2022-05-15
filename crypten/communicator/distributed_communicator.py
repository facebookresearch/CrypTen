#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import random
import string

import numpy
import torch
import torch.distributed as dist
from crypten.common import serial
from torch.distributed import ReduceOp

from .communicator import _logging, Communicator


class DistributedCommunicator(Communicator):
    """
    Implementation of the Communicator class via torch.distributed. Use this
    communicator to communicate between different processes, potentially,
    running on different nodes.
    """

    BYTES_PER_ELEMENT = 8
    instance = None

    def __init__(self, init_ttp=False):
        # no need to do anything if we already initialized the communicator:
        if not dist.is_initialized():
            # get configuration variables from environmens:
            for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
                if key.upper() not in os.environ:
                    raise ValueError("Environment variable %s must be set." % key)
                setattr(self, key.lower(), os.environ[key.upper()])

            # make sure world size and rank are integers; comms stats are reset:
            self.world_size = int(self.world_size)
            self.rank = int(self.rank)
            self.reset_communication_stats()
            self._name = f"rank{self.rank}"

            # logging:
            logging.info("==================")
            logging.info("DistributedCommunicator with rank %d" % self.rank)
            logging.info("==================")

            # initialize process group:
            total_ws = self.world_size + 1 if init_ttp else self.world_size
            dist.init_process_group(
                backend=self.distributed_backend,
                init_method=self.rendezvous,
                world_size=total_ws,
                rank=self.rank,
            )
            self.ttp_group = dist.new_group(list(range(total_ws)))
            if total_ws > 1:
                self.ttp_comm_group = dist.new_group([0, total_ws - 1])
            self.main_group = dist.new_group(list(range(self.world_size)))
            self.ttp_initialized = init_ttp
            logging.info("World size = %d" % self.world_size)

    @classmethod
    def is_initialized(cls):
        if cls.instance is None:
            return False
        return dist.is_initialized()

    @classmethod
    def initialize(cls, rank, world_size, init_ttp=False):
        import os

        if os.name == "nt":
            raise OSError(
                "Multiprocessing is not supported on Windows. "
                + "Please initialize CrypTen via crypten.init_thread() instead."
            )

        # set default arguments for communicator:
        randomized_path = "crypten-".join(
            random.choice(string.ascii_letters) for i in range(10)
        )
        default_args = {
            "DISTRIBUTED_BACKEND": "gloo",
            "RENDEZVOUS": f"file:///tmp/{randomized_path}",
            "WORLD_SIZE": world_size,
            "RANK": rank,
        }
        for key, val in default_args.items():
            if key not in os.environ:
                os.environ[key] = str(val)

        cls.instance = DistributedCommunicator(init_ttp=init_ttp)

    @classmethod
    def get(cls):
        return cls.instance

    @classmethod
    def shutdown(cls):
        if dist.get_rank() == 0 and cls.instance.ttp_initialized:
            cls.instance.send_obj(
                "terminate", cls.instance.get_ttp_rank(), cls.instance.ttp_group
            )
        dist.destroy_process_group(cls.instance.main_group)
        dist.destroy_process_group(cls.instance.ttp_group)
        dist.destroy_process_group()
        cls.instance = None

    @_logging
    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.send(tensor.data, dst, group=self.main_group)

    @_logging
    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.recv(result.data, src=src, group=self.main_group)
        return result

    @_logging
    def isend(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.isend(tensor.data, dst, group=self.main_group)

    @_logging
    def irecv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.irecv(tensor.data, src=src, group=self.main_group)

    @_logging
    def scatter(self, scatter_list, src, size=None, device=None):
        """Scatters a list of tensors to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        if src != self.get_rank():
            if size is None:
                size = scatter_list[self.get_rank()].size()
            if device is None:
                try:
                    device = scatter_list[self.get_rank()].device
                except Exception:
                    pass
            tensor = torch.empty(size=size, dtype=torch.long, device=device)
            dist.scatter(tensor.data, [], src, group=self.main_group)
        else:
            scatter_list = [s.data for s in scatter_list]
            tensor = scatter_list[self.get_rank()]
            dist.scatter(tensor.data, scatter_list, src, group=self.main_group)
        return tensor

    @_logging
    def reduce(self, input, dst, op=ReduceOp.SUM, batched=False):
        """Reduces the input data across all parties."""
        assert dist.is_initialized(), "initialize the communicator first"

        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            reqs = []
            result = [x.clone().data for x in input]
            for tensor in result:
                reqs.append(
                    dist.reduce(
                        tensor.data, dst, op=op, group=self.main_group, async_op=True
                    )
                )
            for req in reqs:
                req.wait()
        else:
            assert torch.is_tensor(
                input.data
            ), "unbatched input for reduce must be a torch tensor"
            result = input.clone()
            dist.reduce(result.data, dst, op=op, group=self.main_group)

        return result if dst == self.get_rank() else None

    @_logging
    def all_reduce(self, input, op=ReduceOp.SUM, batched=False):
        """Reduces the input data across all parties; all get the final result."""
        assert dist.is_initialized(), "initialize the communicator first"

        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            reqs = []
            result = [x.clone() for x in input]
            for tensor in result:
                reqs.append(
                    dist.all_reduce(
                        tensor.data, op=op, group=self.main_group, async_op=True
                    )
                )
            for req in reqs:
                req.wait()
        else:
            assert torch.is_tensor(
                input.data
            ), "unbatched input for reduce must be a torch tensor"
            result = input.clone()
            dist.all_reduce(result.data, op=op, group=self.main_group)
        return result

    @_logging
    def gather(self, tensor, dst):
        """Gathers a list of tensors in a single party."""
        assert dist.is_initialized(), "initialize the communicator first"
        if self.get_rank() == dst:
            result = []
            device = tensor.data.device
            for _ in range(self.get_world_size()):
                result.append(
                    torch.empty(size=tensor.size(), dtype=torch.long, device=device)
                )
            dist.gather(tensor.data, result, dst, group=self.main_group)
            return result
        dist.gather(tensor.data, [], dst, group=self.main_group)
        return [None]

    @_logging
    def all_gather(self, tensor):
        """Gathers tensors from all parties in a list."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = []
        device = tensor.data.device
        for _ in range(self.get_world_size()):
            result.append(
                torch.empty(size=tensor.size(), dtype=torch.long, device=device)
            )
        dist.all_gather(result, tensor.data, group=self.main_group)
        return result

    @_logging
    def broadcast(self, input, src, group=None, batched=False):
        """Broadcasts the tensor to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        group = self.main_group if group is None else group
        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            reqs = []
            for tensor in input:
                reqs.append(
                    dist.broadcast(tensor.data, src, group=group, async_op=True)
                )
            for req in reqs:
                req.wait()
        else:
            assert torch.is_tensor(
                input.data
            ), "unbatched input for reduce must be a torch tensor"
            dist.broadcast(input.data, src, group=group)
        return input

    @_logging
    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        assert dist.is_initialized(), "initialize the communicator first"
        dist.barrier(group=self.main_group)

    @_logging
    def send_obj(self, obj, dst, group=None):
        """Sends the specified object to the destination `dst`."""
        if group is None:
            group = self.main_group

        buf = pickle.dumps(obj)
        size = torch.tensor(len(buf), dtype=torch.int32)
        arr = torch.from_numpy(numpy.copy(numpy.frombuffer(buf, dtype=numpy.int8)))

        r0 = dist.isend(size, dst=dst, group=group)
        r1 = dist.isend(arr, dst=dst, group=group)

        r0.wait()
        r1.wait()

    @_logging
    def recv_obj(self, src, group=None):
        """Receives an object from a source `src`."""
        if group is None:
            group = self.main_group

        size = torch.tensor(1, dtype=torch.int32)
        dist.irecv(size, src=src, group=group).wait()

        data = torch.empty(size=(size,), dtype=torch.int8)
        dist.irecv(data, src=src, group=group).wait()
        buf = data.numpy().tobytes()
        return serial.restricted_loads(buf)

    @_logging
    def broadcast_obj(self, obj, src, group=None):
        """Broadcasts a given object to all parties."""
        if group is None:
            group = self.main_group

        if self.rank == src:
            assert obj is not None, "src party must provide obj for broadcast"
            buf = pickle.dumps(obj)
            size = torch.tensor(len(buf), dtype=torch.int32)
            arr = torch.from_numpy(numpy.copy(numpy.frombuffer(buf, dtype=numpy.int8)))

            dist.broadcast(size, src, group=group)
            dist.broadcast(arr, src, group=group)
        else:
            size = torch.tensor(1, dtype=torch.int32)
            dist.broadcast(size, src, group=group)

            data = torch.empty(size=(size,), dtype=torch.int8)
            dist.broadcast(data, src, group=group)
            buf = data.numpy().tobytes()
            obj = serial.restricted_loads(buf)
        return obj

    def get_world_size(self):
        """Returns the size of the world."""
        assert dist.is_initialized(), "initialize the communicator first"
        return self.world_size

    def get_rank(self):
        """Returns the rank of the current process."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_rank()

    def get_ttp_rank(self):
        """Returns the rank of the Trusted Third Party"""
        return self.get_world_size()

    def set_name(self, name):
        """Sets the party name of the current process."""
        assert isinstance(
            name, str
        ), f"Improper name provided to process on rank {self.get_rank()}"
        self._name = name

    def get_name(self):
        """Returns the party name of the current process."""
        return self._name

    def get_distributed_backend(self):
        """Returns name of torch.distributed backend used."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_backend()
