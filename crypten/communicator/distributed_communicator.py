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

            self.__setup_shared_rng()

    def __setup_shared_rng(self):
        """
            Generate shared random seeds to generate pseudo-random sharings of
            zero. The random seeds are shared such that each process shares
            one seed with the previous rank process and one with the next rank.
            This allows for the generation of `n` random values, each known to
            exactly two of the `n` parties.

            For arithmetic sharing, one of theseparties will add the number
            while the other subtracts it, allowing for the generation of a
            pseudo-random sharing of zero. (This can be done for binary
            sharing using bitwise-xor rather than addition / subtraction)
        """
        # Initialize RNG Generators
        self.g0 = torch.Generator()
        self.g1 = torch.Generator()

        # Generate random seeds for Generators
        # NOTE: Chosen seed can be any number, but it chooses as a random 64-bit
        # integer so other parties cannot guess its value.
        next_seed = torch.randint(-2 ** 63, 2 ** 63 - 1, (1,))
        prev_seed = torch.LongTensor([0])  # placeholder

        # Send random seed to next party, receive random seed from prev party
        if dist.get_world_size() >= 2:  # Otherwise sending seeds will segfault.
            next_rank = (dist.get_rank() + 1) % dist.get_world_size()
            prev_rank = (next_rank - 2) % dist.get_world_size()

            req0 = dist.isend(tensor=next_seed, dst=next_rank)
            req1 = dist.irecv(tensor=prev_seed, src=prev_rank)

            req0.wait()
            req1.wait()
        else:
            prev_seed = next_seed

        # Seed Generators
        self.g0.manual_seed(next_seed.item())
        self.g1.manual_seed(prev_seed.item())

    def shutdown(self):
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

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0

    def print_communication_stats(self):
        """Prints communication statistics."""
        logging.info("====Communication Stats====")
        logging.info("Rounds: %d" % self.comm_rounds)
        logging.info("Bytes : %d" % self.comm_bytes)

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += nelement * self.BYTES_PER_ELEMENT
