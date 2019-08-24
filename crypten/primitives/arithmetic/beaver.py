#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
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
        import crypten

        a, b, c = crypten.TrustedThirdParty.generate_additive_triple(
            x.size(), y.size(), op, *args, **kwargs
        )
        result._tensor = c._tensor

        # Stack to vectorize reveal if possible
        if x.size() == y.size():
            eps_del = crypten.ArithmeticSharedTensor.stack([x - a, y - b]).reveal()
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

        import crypten

        r, r2 = crypten.TrustedThirdParty.square(x.size())
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
        import crypten

        r, theta_r = crypten.TrustedThirdParty.wrap_rng(x.size(), comm.get_world_size())
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
