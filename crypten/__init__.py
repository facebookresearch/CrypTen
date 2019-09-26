#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401
import torch

# other imports:
from . import debug
from .cryptensor import CrypTensor
from .mpc import ptype


def init():
    comm._init(use_threads=False)
    _setup_przs()


def init_thread(rank, world_size):
    comm._init(use_threads=True, rank=rank, world_size=world_size)
    _setup_przs()


def uninit():
    return comm.uninit()


# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary


def print_communication_stats():
    comm.get().print_communication_stats()


def reset_communication_stats():
    comm.get().reset_communication_stats()


# Set backend
__SUPPORTED_BACKENDS = [crypten.mpc]
__default_backend = __SUPPORTED_BACKENDS[0]


def set_default_backend(new_default_backend):
    """Sets the default cryptensor backend (mpc, he)"""
    global __default_backend
    assert new_default_backend in __SUPPORTED_BACKENDS, (
        "Backend %s is not supported" % new_default_backend
    )
    __default_backend = new_default_backend


def get_default_backend():
    """Returns the default cryptensor backend (mpc, he)"""
    return __default_backend


def cryptensor(*args, backend=None, **kwargs):
    """
    Factory function to return encrypted tensor of given backend.
    """
    if backend is None:
        backend = get_default_backend()
    if backend == crypten.mpc:
        return backend.MPCTensor(*args, **kwargs)
    else:
        raise TypeError("Backend %s is not supported" % backend)


def is_encrypted_tensor(obj):
    """
    Returns True if obj is an encrypted tensor.
    """
    return isinstance(obj, CrypTensor)


def _setup_przs():
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
    comm.get().g0 = torch.Generator()
    comm.get().g1 = torch.Generator()

    # Generate random seeds for Generators
    # NOTE: Chosen seed can be any number, but we choose as a random 64-bit
    # integer here so other parties cannot guess its value.

    # We sometimes get here from a forked process, which causes all parties
    # to have the same RNG state. Reset the seed to make sure RNG streams
    # are different in all the parties. We use numpy's random here since
    # setting its seed to None will produce different seeds even from
    # forked processes.
    import numpy

    numpy.random.seed(seed=None)
    next_seed = torch.tensor(numpy.random.randint(-2 ** 63, 2 ** 63 - 1, (1,)))
    prev_seed = torch.LongTensor([0])  # placeholder

    # Send random seed to next party, receive random seed from prev party
    world_size = comm.get().get_world_size()
    rank = comm.get().get_rank()
    if world_size >= 2:  # Otherwise sending seeds will segfault.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        req0 = comm.get().isend(tensor=next_seed, dst=next_rank)
        req1 = comm.get().irecv(tensor=prev_seed, src=prev_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed

    # Seed Generators
    comm.get().g0.manual_seed(next_seed.item())
    comm.get().g1.manual_seed(prev_seed.item())


def load(f, encrypted=False, src=None, **kwargs):
    """
    Loads an object saved with `torch.save()` or `crypten.save()`.

    Parameters:
        `f` - a file-like object (has to implement read(), :meth`readline`,
              :meth`tell`, and :meth`seek`), or a string containing a file name
        `encrypted` - Determines whether crypten should load an encrypted tesnor
                      or a plaintext torch tensor.
        `src` - Determines the source of the tensor. If `src` is None, each
                party will attempt to read in the specified file. If `src` is
                specified, the source party will read the tensor from
    """
    if encrypted:
        raise NotImplementedError("Loading encrypted tensors is not yet supported")
    else:
        if src is None:
            return torch.load(f, **kwargs)
        else:
            assert isinstance(src, int), "Load failed: src argument must be an integer"
            assert (
                src >= 0 and src < comm.get().get_world_size()
            ), "Load failed: src must be in [0, world_size)"

            if comm.get().get_rank() == src:
                result = torch.load(f, **kwargs)

                # Broadcast size to other parties.
                dim = torch.tensor(result.dim(), dtype=torch.long)
                size = torch.tensor(result.size(), dtype=torch.long)

                comm.get().broadcast(dim, src=src)
                comm.get().broadcast(size, src=src)

            else:
                # Receive size from source party
                dim = torch.empty(size=(), dtype=torch.long)
                comm.get().broadcast(dim, src=src)
                size = torch.empty(size=(dim.item(),), dtype=torch.long)
                comm.get().broadcast(size, src=src)
                result = torch.empty(size=tuple(size.tolist()))

            return result


def save(obj, f, src=0, **kwargs):
    """
    Saves a CrypTensor or PyTorch tensor to a file.

    Parameters:
        `obj` - The CrypTensor or PyTorch tensor to be saved
        `f` - a file-like object (has to implement write and flush) or a string
              containing a file name
        `src` - The source party that writes data to the specified file.
    """
    if is_encrypted_tensor(obj):
        raise NotImplementedError("Saving encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Save failed: src must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Save failed: src must be an integer in [0, world_size)"

        if comm.get().get_rank() == src:
            torch.save(obj, f, **kwargs)

    # Implement barrier to avoid race conditions that require file to exist
    comm.get().barrier()


def where(condition, input, other):
    """
    Return a tensor of elements selected from either `input` or `other`, depending
    on `condition`.
    """
    if is_encrypted_tensor(condition):
        return condition * input + (1 - condition) * other
    elif torch.is_tensor(condition):
        condition = condition.float()
    return input * condition + other * (1 - condition)


# Top level tensor functions
__PASSTHROUGH_FUNCTIONS = ["bernoulli", "cat", "rand", "randperm", "stack"]


def __add_top_level_function(func_name):
    def _passthrough_function(*args, backend=None, **kwargs):
        if backend is None:
            backend = get_default_backend()
        return getattr(backend, func_name)(*args, **kwargs)

    globals()[func_name] = _passthrough_function


for func in __PASSTHROUGH_FUNCTIONS:
    __add_top_level_function(func)

# expose classes and functions in package:
__all__ = ["CrypTensor", "debug", "init", "init_thread", "mpc", "nn", "uninit"]
