#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import crypten
from crypten import comm
from crypten.common.util import count_wraps


class Beaver:
    """
    Beaver class contains all protocols that make calls to the TTP to
    perform variable hiding
    """

    @staticmethod
    def __beaver_protocol(op, x, y, *args, **kwargs):
        """Performs Beaver protocol for additively secret-shared tensors x and y

        1. Additively hide x and y with appropriately sized a and b
        2. Open (epsilon = x - a) and (delta = y - b)
        3. Return z = epsilon * delta + epsilon * b + a * delta + (c = a * b)
        """
        # Clone the input to make result's attributes consistent with the input
        result = x.shallow_copy()

        assert op in ["mul", "matmul", "conv2d"]

        provider = crypten.get_default_provider()
        a, b, c = provider.generate_additive_triple(
            x.size(), y.size(), op, *args, **kwargs
        )
        result._tensor = c._tensor

        # Stack to vectorize reveal if possible
        if x.size() == y.size():
            from .arithmetic import ArithmeticSharedTensor
            eps_del = ArithmeticSharedTensor.stack([x - a, y - b]).reveal()
            epsilon = eps_del[0]
            delta = eps_del[1]
        else:
            epsilon = (x - a).reveal()
            delta = (y - b).reveal()

        result._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
        result._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
        if result.rank == 0:
            result._tensor += getattr(torch, op)(epsilon, delta, *args, **kwargs)

        return c

    @staticmethod
    def mul(x, y):
        return Beaver.__beaver_protocol("mul", x, y)

    @staticmethod
    def matmul(x, y):
        return Beaver.__beaver_protocol("matmul", x, y)

    @staticmethod
    def conv2d(x, y, **kwargs):
        return Beaver.__beaver_protocol("conv2d", x, y, **kwargs)

    @staticmethod
    def square(x):
        # Clone the input to make result's attributes consistent with the input
        result = x.clone()

        provider = crypten.get_default_provider()
        r, r2 = provider.square(x.size())
        result._tensor = r2._tensor

        epsilon = (x - r).reveal()
        result._tensor += r._tensor.mul_(epsilon).mul_(2)
        if result.rank == 0:
            result._tensor += epsilon.mul_(epsilon)

        return result

    @staticmethod
    def wraps(x):
        """Privately computes the number of wraparounds for a set a shares

        To do so, we note that:
            [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

        Where [theta_i] is the wraps for a variable i
              [beta_ij] is the differential wraps for variables i and j
              [eta_ij]  is the plaintext wraps for variables i and j

        Since [eta_xr] = 0 with probability |x| / Q for modulus Q, we can make
        the assumption that [eta_xr] = 0 with high probability.
        """
        provider = crypten.get_default_provider()
        r, theta_r = provider.wrap_rng(x.size(), comm.get_world_size())
        beta_xr = theta_r.clone()
        beta_xr._tensor = count_wraps([x._tensor, r._tensor])

        z = x + r
        theta_z = comm.gather(z._tensor, 0)
        theta_x = beta_xr - theta_r

        # TODO: Incorporate eta_xr
        if x.rank == 0:
            theta_z = count_wraps(theta_z)
            theta_x._tensor += theta_z
        return theta_x

    @staticmethod
    def AND(x, y):
        """Performs Beaver AND protocol for BinarySharedTensor tensors x and y"""
        from .binary import BinarySharedTensor

        provider = crypten.get_default_provider()
        a, b, c = provider.generate_xor_triple(x.size())

        # Stack to vectorize reveal
        eps_del = BinarySharedTensor.stack([x ^ a, y ^ b]).reveal()
        epsilon = eps_del[0]
        delta = eps_del[1]

        c._tensor ^= (epsilon & b._tensor) ^ (a._tensor & delta)
        if c.rank == 0:
            c._tensor ^= epsilon & delta

        return c

    @staticmethod
    def B2A_single_bit(xB):
        """Converts a single-bit BinarySharedTensor xB into an
            ArithmeticSharedTensor. This is done by:

        1. Generate ArithmeticSharedTensor [rA] and BinarySharedTensor =rB= with
            a common 1-bit value r.
        2. Hide xB with rB and open xB ^ rB
        3. If xB ^ rB = 0, then return [rA], otherwise return 1 - [rA]
            Note: This is an arithmetic xor of a single bit.
        """
        if comm.get_world_size() < 2:
            from crypten.primitives import ArithmeticSharedTensor
            return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

        provider = crypten.get_default_provider()
        rA, rB = provider.B2A_rng(xB.size())

        z = (xB ^ rB).reveal()
        rA._tensor = rA._tensor * (1 - z) - rA._tensor * z
        if rA.rank == 0:
            rA._tensor += z
        return rA
