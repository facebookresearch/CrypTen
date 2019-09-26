#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from .communicator import Communicator, _logging


class DistributedCommunicator(Communicator):
    """
    Implementation of the Communicator class via torch.distributed. Use this
    communicator to communicate between different processes, potentially,
    running on different nodes.
    """

    BYTES_PER_ELEMENT = 8
    __instance = None

    def __init__(self):
        # no need to do anything if we already initialized the communicator:
        if not dist.is_initialized():
            # get configuration variables from environmens:
            state = {}
            for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
                key = key.upper()
                if key not in os.environ:
                    raise ValueError("Environment variable %s must be set." % key)
                state[key.lower()] = os.environ[key]

            # make sure world size and rank are integers; comms stats are reset:
            state["world_size"] = int(state["world_size"])
            state["rank"] = int(state["rank"])
            self.reset_communication_stats()

            # logging:
            logging.info("==================")
            logging.info("DistributedCommunicator with rank %d" % state["rank"])
            logging.info("==================")

            # initialize process group:
            dist.init_process_group(
                backend=state["distributed_backend"],
                init_method=state["rendezvous"],
                world_size=state["world_size"],
                rank=state["rank"],
            )
            logging.info("World size = %d" % dist.get_world_size())

    @classmethod
    def is_initialized(cls):
        return dist.is_initialized()

    @classmethod
    def initialize(cls, rank, world_size):
        import os

        if os.name == "nt":
            raise OSError(
                "Multiprocessing is not supported on Windows. "
                + "Please initialize CrypTen via crypten.init_thread() instead."
            )

        # set default arguments for communicator:
        default_args = {
            "DISTRIBUTED_BACKEND": "gloo",
            "RENDEZVOUS": "file:///tmp/sharedfile",
            "WORLD_SIZE": world_size,
            "RANK": rank,
        }
        for key, val in default_args.items():
            if key not in os.environ:
                os.environ[key] = str(val)

        cls.instance = DistributedCommunicator()

    @classmethod
    def get(cls):
        return cls.instance

    @classmethod
    def shutdown(cls):
        dist.destroy_process_group()

    @_logging
    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.send(tensor, dst)

    @_logging
    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.recv(result, src=src)
        return result

    @_logging
    def isend(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.isend(tensor, dst)

    @_logging
    def irecv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.irecv(tensor, src=src)

    @_logging
    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of tensors to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        if src != self.get_rank():
            if size is None:
                size = scatter_list[self.get_rank()].size()
            tensor = torch.empty(size=size, dtype=torch.long)
            dist.scatter(tensor, [], src, async_op=async_op)
        else:
            tensor = scatter_list[self.get_rank()]
            dist.scatter(tensor, [t for t in scatter_list], src, async_op=async_op)
        return tensor

    @_logging
    def reduce(self, tensor, dst, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.reduce(result, dst, op=op, async_op=async_op)
        return result

    @_logging
    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.all_reduce(result, op=op, async_op=async_op)
        return result

    @_logging
    def gather(self, tensor, dst, async_op=False):
        """Gathers a list of tensors in a single party."""
        assert dist.is_initialized(), "initialize the communicator first"
        if self.get_rank() == dst:
            result = []
            for _ in range(self.get_world_size()):
                result.append(torch.empty(size=tensor.size(), dtype=torch.long))
            dist.gather(tensor, result, dst, async_op=async_op)
            return result
        dist.gather(tensor, [], dst, async_op=async_op)

    @_logging
    def all_gather(self, tensor, async_op=False):
        """Gathers tensors from all parties in a list."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = []
        for _ in range(self.get_world_size()):
            result.append(torch.empty(size=tensor.size(), dtype=torch.long))
        dist.all_gather(result, tensor, async_op=async_op)
        return result

    @_logging
    def broadcast(self, tensor, src, async_op=False):
        """Broadcasts the tensor to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.broadcast(tensor, src, async_op=async_op)
        return tensor

    @_logging
    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        assert dist.is_initialized(), "initialize the communicator first"
        dist.barrier()

    def get_world_size(self):
        """Returns the size of the world."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_world_size()

    def get_rank(self):
        """Returns the rank of the current process."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_rank()

    def get_distributed_backend(self):
        """Returns name of torch.distributed backend used."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_backend()
