#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch
from crypten.common.util import count_wraps


def __beaver_protocol(op, x, y, *args, **kwargs):
    """Performs Beaver protocol for additively secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a * b]
    2. Additively hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] - [a]) and ([delta] = [y] - [b])
    4. Return [z] = [c] + (epsilon * [b]) + ([a] * delta) + (epsilon * delta)
    """
    assert op in ["mul", "matmul", "conv2d", "conv_transpose2d"]

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_additive_triple(x.size(), y.size(), op, *args, **kwargs)

    # Stack to vectorize reveal if possible
    if x.size() == y.size():
        from .arithmetic import ArithmeticSharedTensor

        eps_del = ArithmeticSharedTensor.stack([x - a, y - b]).reveal()
        epsilon = eps_del[0]
        delta = eps_del[1]
    else:
        epsilon = (x - a).reveal()
        delta = (y - b).reveal()

    # z = c + (a * delta) + (epsilon * b) + epsilon * delta
    # TODO: Implement crypten.mul / crypten.matmul / crypten.conv{_transpose}2d
    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    c += getattr(a, op)(delta, *args, **kwargs)
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)

    return c


def mul(x, y):
    return __beaver_protocol("mul", x, y)


def matmul(x, y):
    return __beaver_protocol("matmul", x, y)


def conv2d(x, y, **kwargs):
    return __beaver_protocol("conv2d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    """Computes the square of `x` for additively secret-shared tensor `x`

    1. Obtain uniformly random sharings [r] and [r2] = [r * r]
    2. Additively hide [x] with appropriately sized [r]
    3. Open ([epsilon] = [x] - [r])
    4. Return z = [r2] + 2 * epsilon * [r] + epsilon ** 2
    """
    provider = crypten.mpc.get_default_provider()
    r, r2 = provider.square(x.size())

    epsilon = (x - r).reveal()
    return r2 + 2 * r * epsilon + epsilon * epsilon


def wraps(x):
    """Privately computes the number of wraparounds for a set a shares

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.
    """
    provider = crypten.mpc.get_default_provider()
    r, theta_r = provider.wrap_rng(x.size())
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    # TODO: Incorporate eta_xr
    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def AND(x, y):
    """
    Performs Beaver protocol for binary secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a & b]
    2. XOR hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] ^ [a]) and ([delta] = [y] ^ [b])
    4. Return [c] ^ (epsilon & [b]) ^ ([a] & delta) ^ (epsilon & delta)
    """
    from .binary import BinarySharedTensor

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size())

    # Stack to vectorize reveal
    eps_del = BinarySharedTensor.stack([x ^ a, y ^ b]).reveal()
    epsilon = eps_del[0]
    delta = eps_del[1]

    return (b & epsilon) ^ (a & delta) ^ (epsilon & delta) ^ c


def B2A_single_bit(xB):
    """Converts a single-bit BinarySharedTensor xB into an
        ArithmeticSharedTensor. This is done by:

    1. Generate ArithmeticSharedTensor [rA] and BinarySharedTensor =rB= with
        a common 1-bit value r.
    2. Hide xB with rB and open xB ^ rB
    3. If xB ^ rB = 0, then return [rA], otherwise return 1 - [rA]
        Note: This is an arithmetic xor of a single bit.
    """
    if comm.get().get_world_size() < 2:
        from .arithmetic import ArithmeticSharedTensor

        return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

    provider = crypten.mpc.get_default_provider()
    rA, rB = provider.B2A_rng(xB.size())

    z = (xB ^ rB).reveal()
    rA = rA * (1 - 2 * z) + z
    return rA
