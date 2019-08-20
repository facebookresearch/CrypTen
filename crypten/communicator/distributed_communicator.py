#!/usr/bin/env python3

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

    def __init__(self):

        # get configuration variables from environmens:
        self.state = {}
        for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
            key = key.upper()
            if key not in os.environ:
                raise ValueError("Environment variable %s must be set." % key)
            self.state[key.lower()] = os.environ[key]

        # make sure world size and rank are integers; comms stats are reset:
        self.state["world_size"] = int(self.state["world_size"])
        self.state["rank"] = int(self.state["rank"])
        self.reset_communication_stats()

        # no need to do anything if we already initialized the communicator:
        if not dist.is_initialized():

            # logging:
            level = logging.getLogger().level
            logging.getLogger().setLevel(logging.INFO)
            logging.info("==================")
            logging.info("DistributedCommunicator with rank %d" % self.state["rank"])
            logging.info("==================")
            logging.getLogger().setLevel(level)

            # initialize process group:
            dist.init_process_group(
                backend=self.state["distributed_backend"],
                init_method=self.state["rendezvous"],
                world_size=self.state["world_size"],
                rank=self.state["rank"],
            )
            logging.info("World size = %d" % self.state["world_size"])

    @_logging
    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.send(tensor, dst)

    @_logging
    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.recv(tensor, src=src)
        return tensor

    @_logging
    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of tensors to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        if src != self.state["rank"]:
            if size is None:
                size = scatter_list[self.state["rank"]].size()
            tensor = torch.empty(size=size, dtype=torch.long)
            dist.scatter(tensor, [], src, async_op=async_op)
        else:
            tensor = scatter_list[self.state["rank"]]
            dist.scatter(tensor, [t for t in scatter_list], src, async_op=async_op)
        return tensor

    @_logging
    def reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.reduce(tensor, op=op, async_op=async_op)
        return tensor

    @_logging
    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.all_reduce(tensor, op=op, async_op=async_op)
        return tensor

    @_logging
    def gather(self, tensor, dst, async_op=False):
        """Gathers a list of tensors in a single party."""
        assert dist.is_initialized(), "initialize the communicator first"
        if self.state["rank"] == dst:
            result = []
            for _ in range(self.state["world_size"]):
                result.append(torch.empty(size=tensor.size(), dtype=torch.long))
            dist.gather(tensor, result, dst, async_op=async_op)
            return result
        dist.gather(tensor, [], dst, async_op=async_op)

    @_logging
    def all_gather(self, tensor, async_op=False):
        """Gathers tensors from all parties in a list."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = []
        for _ in range(self.state["world_size"]):
            result.append(torch.empty(size=tensor.size(), dtype=torch.long))
        dist.all_gather(result, tensor, async_op=async_op)
        return result

    @_logging
    def broadcast(self, tensor, src, async_op=False):
        """Broadcasts the tensor to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.broadcast(tensor, src, async_op=async_op)
        return tensor

    def get_world_size(self):
        """Returns the size of the world."""
        assert dist.is_initialized(), "initialize the communicator first"
        return self.state["world_size"]

    def get_rank(self):
        """Returns the rank of the current process."""
        assert dist.is_initialized(), "initialize the communicator first"
        return self.state["rank"]

    def get_distributed_backend(self):
        """Returns name of torch.distributed backend used."""
        assert dist.is_initialized(), "initialize the communicator first"
        return self.state["distributed_backend"]

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.state["comm_rounds"] = 0
        self.state["comm_bytes"] = 0

    def print_communication_stats(self):
        """Prints communication statistics."""
        logging.info("====Communication Stats====")
        logging.info("Rounds: %d" % self.state["comm_rounds"])
        logging.info("Bytes : %d" % self.state["comm_bytes"])

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.state["comm_rounds"] += 1
        self.state["comm_bytes"] += nelement * self.BYTES_PER_ELEMENT
