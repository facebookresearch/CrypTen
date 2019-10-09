#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor


class TrustedThirdParty:
    @staticmethod
    def generate_additive_triple(size0, size1, op, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        a = generate_random_ring_element(size0)
        b = generate_random_ring_element(size1)
        c = getattr(torch, op)(a, b, *args, **kwargs)

        a = ArithmeticSharedTensor(a, precision=0, src=0)
        b = ArithmeticSharedTensor(b, precision=0, src=0)
        c = ArithmeticSharedTensor(c, precision=0, src=0)

        return a, b, c

    @staticmethod
    def square(size):
        """Generate square double of given size"""
        r = generate_random_ring_element(size)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        stacked = torch.stack([r, r2])
        stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    @staticmethod
    def generate_xor_triple(size):
        """Generate xor triples of given size"""
        a = generate_kbit_random_tensor(size)
        b = generate_kbit_random_tensor(size)
        c = a & b

        # Stack to vectorize scatter function
        abc = torch.stack([a, b, c])
        abc = BinarySharedTensor(abc, src=0)
        return abc[0], abc[1], abc[2]

    @staticmethod
    def wrap_rng(size, num_parties):
        """Generate random shared tensor of given size and sharing of its wraps"""
        r = [generate_random_ring_element(size) for _ in range(num_parties)]
        theta_r = count_wraps(r)

        shares = comm.get().scatter(r, src=0)
        r = ArithmeticSharedTensor.from_shares(shares, precision=0)
        theta_r = ArithmeticSharedTensor(theta_r, precision=0, src=0)

        return r, theta_r

    @staticmethod
    def B2A_rng(size):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        # generate random bit
        r = generate_kbit_random_tensor(size, bitlength=1)

        rA = ArithmeticSharedTensor(r, precision=0, src=0)
        rB = BinarySharedTensor(r, src=0)

        return rA, rB

    @staticmethod
    def rand(*sizes):
        """Generate random ArithmeticSharedTensor uniform on [0, 1]"""
        samples = torch.rand(*sizes)
        return ArithmeticSharedTensor(samples, src=0)

    @staticmethod
    def bernoulli(tensor):
        """Generate random ArithmeticSharedTensor bernoulli on {0, 1}"""
        samples = torch.bernoulli(tensor)
        return ArithmeticSharedTensor(samples, src=0)

    @staticmethod
    def randperm(tensor_size):
        """
        Generate `tensor_size[:-1]` random ArithmeticSharedTensor permutations of
        the first `tensor_size[-1]` whole numbers
        """
        tensor_len = tensor_size[-1]
        nperms = int(torch.tensor(tensor_size[:-1]).prod().item())
        random_permutation = (
            torch.stack([torch.randperm(tensor_len) + 1 for _ in range(nperms)])
            .view(tensor_size)
            .float()
        )
        return ArithmeticSharedTensor(random_permutation, src=0)
