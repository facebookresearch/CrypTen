#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch

from crypten.cuda import CUDALongTensor
from crypten.common.util import torch_stack
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.encoder import FixedPointEncoder

import time

def replicate_shares(x_share):
    """
    Replicate additively secret-shared tensor x。
    Party i sends x_i to party i+1, and receives
    x_{i-1} from party i-1.
    Each party holds 2-out-of-3 secret sharing of tensor
    x after the protocol. 
    """
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    x_share = x_share.contiguous()
    x_rep = torch.zeros_like(x_share.data)

    req1 = comm.get().isend(x_share.data, dst=next_rank)
    req2 = comm.get().irecv(x_rep.data, src=prev_rank)

    req1.wait()
    req2.wait()

    return x_rep


def __replicated_secret_sharing_protocol(op, x, y, *args, **kwargs):
    """
    Implement Replicated Secret Sharing (RSS) protocol for 3PC computation.
    Each party holds 2-out-of-3 secret sharing of tnesor x and y. 
    See Section 3.2.1 of "ABY3 : A Mixed Protocol Framework for Machine Learning"
    for more detail.
    """
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor

    x1, x2 = x.share, replicate_shares(x.share)
    y1, y2 = y.share, replicate_shares(y.share)

    z = getattr(torch, op)(x1, y1, *args, **kwargs) + getattr(torch, op)(x1, y2, *args, **kwargs) + getattr(torch, op)(x2, y1, *args, **kwargs)


    rank = comm.get().get_rank()
    if isinstance(x, BinarySharedTensor):
        z = BinarySharedTensor.from_shares(z, src=rank)
        z += BinarySharedTensor.PRZS(z.size(), device=z.device)
    elif isinstance(x, ArithmeticSharedTensor):
        z = ArithmeticSharedTensor.from_shares(z, src=rank)
        z += ArithmeticSharedTensor.PRZS(z.size(), device=z.device)

    return z


def mul(x, y):
    return __replicated_secret_sharing_protocol("mul", x, y)


def matmul(x, y):
    return __replicated_secret_sharing_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    from .arithmetic import ArithmeticSharedTensor
    
    x1 = x.share
    x2 = replicate_shares(x.share)
    x_square = x1 ** 2 + 2 * x1 * x2
    
    x_square = ArithmeticSharedTensor.from_shares(x_square, src=comm.get().get_rank())
    x_square += ArithmeticSharedTensor.PRZS(x_square.size(), device=x_square.device)
    return x_square


def truncation(x, scale):
    """
    Performs three-party share truncation protocol
    (See Figure 2 of "ABY3 : A Mixed Protocol Framework for Machine Learning")
    """
    rank = x.rank
    rep_share = replicate_shares(x.share)
    
    if rank == 0:
        r = generate_random_ring_element(x.share.size(), device=x.device, generator=comm.get().get_generator(0, device=x.device))
        x.share = r
    if rank == 1:
        r = generate_random_ring_element(x.share.size(), device=x.device, generator=comm.get().get_generator(1, device=x.device))
        x.share = (x.share + rep_share) // scale - r
    if rank == 2:
        x.share //= scale

    return x


def AND(x, y):
    from .binary import BinarySharedTensor
    
    x1, x2 = x.share, replicate_shares(x.share)
    y1, y2 = y.share, replicate_shares(y.share)

    z = BinarySharedTensor.from_shares((x1 & y1) ^ (x2 & y1) ^ (x1 & y2), src=comm.get().get_rank())
    z += BinarySharedTensor.PRZS(z.size(), device=z.device)

    return z