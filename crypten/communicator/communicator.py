#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import timeit

import torch


class Communicator:
    """
    Abstract class defining the functions that a Communicator should implement.
    """

    # Determines whether communicators log communication stats
    __verbosity = False

    @classmethod
    def is_verbose(cls):
        return cls.__verbosity

    @classmethod
    def set_verbosity(cls, verbosity):
        assert isinstance(verbosity, bool), "Verbosity must be a boolean value"
        cls.__verbosity = verbosity

    @classmethod
    def is_initialized(cls):
        """Returns whether the communicator has been initialized"""
        raise NotImplementedError("is_initialized is not implemented")

    @classmethod
    def get(cls):
        """Returns an instance of the communicator"""
        raise NotImplementedError("get is not implemented")

    @classmethod
    def initialize(cls, **kwargs):
        """Initializes the communicator. Call this function before using it."""
        raise NotImplementedError("initialize is not implemented")

    @classmethod
    def shutdown(cls):
        raise NotImplementedError("shutdown is not implemented")

    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        raise NotImplementedError("send is not implemented")

    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        raise NotImplementedError("recv is not implemented")

    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of tensors to all parties."""
        raise NotImplementedError("scatter is not implemented")

    def reduce(self, tensor, op=None, async_op=False):
        """Reduces the tensor data across all parties."""
        raise NotImplementedError("tensor is not implemented")

    def all_reduce(self, tensor, op=None, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        raise NotImplementedError("tensor is not implemented")

    def gather(self, tensor, dst, async_op=False):
        """Gathers a list of tensors in a single party."""
        raise NotImplementedError("gather is not implemented")

    def all_gather(self, tensor, async_op=False):
        """Gathers tensors from all parties in a list."""
        raise NotImplementedError("all_gather is not implemented")

    def broadcast(self, tensor, src, async_op=False):
        """Broadcasts the tensor to all parties."""
        raise NotImplementedError("broadcast is not implemented")

    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        raise NotImplementedError("barrier is not implemented")

    def send_obj(self, obj, dst):
        """Sends the specified object to the destination `dst`."""
        raise NotImplementedError("send_obj is not implemented")

    def recv_obj(self, src):
        """Receives a tensor from a source src."""
        raise NotImplementedError("recv_obj is not implemented")

    def broadcast_obj(self, obj, src):
        """Broadcasts a given object to all parties."""
        raise NotImplementedError("broadcast_obj is not implemented")

    def get_world_size(self):
        """Returns the size of the world."""
        raise NotImplementedError("get_world_size is not implemented")

    def get_rank(self):
        """Returns the rank of the current process."""
        raise NotImplementedError("get_rank is not implemented")

    def set_name(self):
        """Sets the party name of the current process."""
        raise NotImplementedError("set_name is not implemented")

    def get_name(self):
        """Returns the party name of the current process."""
        raise NotImplementedError("get_name is not implemented")

    def reset_communication_stats(self):
        """Resets communication statistics."""
        raise NotImplementedError("reset_communication_stats is not implemented")

    def print_communication_stats(self):
        """Prints communication statistics."""
        raise NotImplementedError("print_communication_stats is not implemented")

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        raise NotImplementedError("_log_communication is not implemented")

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_time = 0

    def print_communication_stats(self):
        """Prints communication statistics."""
        logging.info("====Communication Stats====")
        logging.info("Rounds: {}".format(self.comm_rounds))
        logging.info("Bytes : {}".format(self.comm_bytes))
        logging.info("Comm time: {}".format(self.comm_time))

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += nelement * self.BYTES_PER_ELEMENT

    def _log_communication_time(self, comm_time):
        self.comm_time += comm_time

    def get_g0(self, device=None):
        assert hasattr(
            self, "g0"
        ), "Generator g0 is not initialized, call crypten.init() first"
        assert device is None or isinstance(
            device, (str, torch.device)
        ), "device must either be a string or a torch.device"

        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (
            isinstance(device, str) and device.startswith("cuda")
        )
        if is_cuda:
            assert (
                torch.cuda.is_available()
            ), "Specifying a GPU device but cuda is not available"
            assert hasattr(
                self, "g0_cuda"
            ), "Generator g0_cuda is not initialized, call crypten.init() first"
            return self.g0_cuda

        return self.g0

    def get_g1(self, device=None):
        assert hasattr(
            self, "g1"
        ), "Generator g1 is not initialized, call crypten.init() first"
        assert device is None or isinstance(
            device, (str, torch.device)
        ), "device must either be a string or a torch.device"

        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (
            isinstance(device, str) and device.startswith("cuda")
        )
        if is_cuda:
            assert (
                torch.cuda.is_available()
            ), "Specifying a GPU device but cuda is not available"
            assert hasattr(
                self, "g1_cuda"
            ), "Generator g1_cuda is not initialized, call crypten.init() first"
            return self.g1_cuda

        return self.g1


def _logging(func):
    """Decorator that performs logging of communication statistics."""

    def logging_wrapper(self, *args, **kwargs):

        # TODO: Replace this
        # - hacks the inputs into some of the functions for world_size 1:
        if self.get_world_size() < 2:
            if func.__name__ in ["gather", "all_gather"]:
                return [args[0]]
            elif len(args) > 0:
                return args[0]

        # only log if needed:
        if self.is_verbose():
            if func.__name__ == "barrier":
                self._log_communication(0, 1)
            elif func.__name__ == "scatter":  # N - 1 tensors communicated
                self._log_communication(args[0][0].nelement() * (len(args[0]) - 1))
            elif "batched" in kwargs and kwargs["batched"]:
                nbytes = sum(x.nelement() for x in args[0])
                self._log_communication(nbytes)
            else:  # one tensor communicated
                self._log_communication(args[0].nelement())

            tic = timeit.timeit()
            result = func(self, *args, **kwargs)
            toc = timeit.timeit()

            self._log_communication_time(toc - tic)
            return result

        return func(self, *args, **kwargs)

    return logging_wrapper
