#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def communicator():
    """Returns distributed communicator."""
    import os
    from .communicator import DistributedCommunicator

    # set default arguments for communicator:
    default_args = {
        "DISTRIBUTED_BACKEND": "gloo",
        "RENDEZVOUS": "file:///tmp/sharedfile",
        "WORLD_SIZE": 1,
        "RANK": 0,
    }
    for key, val in default_args.items():
        if key not in os.environ:
            os.environ[key] = str(val)

    # return communicator:
    return DistributedCommunicator()


# initialize communicator:
comm = communicator()

import crypten.nn  # noqa: F401
import crypten.primitives  # noqa: F401
import crypten.provider  # noqa: F401
import torch

# other imports:
from .common.encrypted_tensor import EncryptedTensor
from .mpc import MPCTensor
from .multiprocessing_pdb import pdb
from .ptype import ptype


# the different private type attributes of an encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary

# expose classes and functions in package:
__all__ = [
    "MPCTensor",
    "EncryptedTensor",
    "primitives",
    "pdb",
    "provider",
    "nn"
]


def __cat_stack_helper(op, tensors, *args, **kwargs):
    assert op in ["cat", "stack"], "Unsupported op for helper function"
    assert isinstance(tensors, list), "%s input must be a list" % op
    assert len(tensors) > 0, "expected a non-empty list of MPCTensors"

    ptype = kwargs.pop("ptype", None)
    # Populate ptype field
    if ptype is None:
        for tensor in tensors:
            if isinstance(tensor, MPCTensor):
                ptype = tensor.ptype
                break
    if ptype is None:
        ptype = arithmetic

    # Make all inputs MPCTensors of given ptype
    for i, tensor in enumerate(tensors):
        if torch.is_tensor(tensor):
            tensors[i] = MPCTensor(tensor, ptype=ptype)
        assert isinstance(tensors[i], MPCTensor), "Can't %s %s with MPCTensor" % (
            op, type(tensor),
        )
        if tensors[i].ptype != ptype:
            tensors[i] = tensors[i].to(ptype)

    # Operate on all input tensors
    result = tensors[0].clone()
    result._tensor._tensor = getattr(torch, op)(
        [tensor._tensor._tensor for tensor in tensors], *args, **kwargs
    )
    return result


def cat(tensors, *args, **kwargs):
    """Perform matrix concatenation"""
    return __cat_stack_helper("cat", tensors, *args, **kwargs)


def stack(tensors, *args, **kwargs):
    """Perform tensor stacking"""
    return __cat_stack_helper("stack", tensors, *args, **kwargs)


def rand(*sizes):
    """
    Returns a tensor with elements uniformly sampled in [0, 1) using the
    trusted third party.
    """
    rand = MPCTensor(None)
    rand._tensor = crypten.provider.TrustedThirdParty.rand(*sizes)
    rand.ptype = arithmetic
    return rand


def bernoulli(tensor):
    """
    Returns a tensor with elements in {0, 1}. The i-th element of the
    output will be 1 with probability according to the i-th value of the
    input tensor.
    """
    return rand(tensor.size()) < tensor


def randperm(size):
    """
        Generate an MPCTensor with rows that contain values [1, 2, ... n]
        where `n` is the length of each row (size[-1])
    """
    result = MPCTensor(None)
    result._tensor = crypten.provider.TrustedThirdParty.randperm(size)
    result.ptype = arithmetic
    return result


def print_communication_stats():
    comm.print_communication_stats()


def reset_communication_stats():
    comm.reset_communication_stats()


# Set provider
__SUPPORTRED_PROVIDERS = [
    crypten.provider.TrustedThirdParty,
    crypten.provider.HomomorphicProvider,
]
__default_provider = __SUPPORTRED_PROVIDERS[0]


def set_default_provider(new_default_provider):
    global __default_provider
    assert new_default_provider in __SUPPORTRED_PROVIDERS, \
        "Provider %s is not supported" % new_default_provider
    __default_provider = new_default_provider


def get_default_provider():
    return __default_provider
