#!/usr/bin/env python3
import torch
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.sharing import share
from crypten.common.util import count_wraps


class TrustedThirdParty:
    @staticmethod
    def generate_additive_triple(size0, size1, op, *args, **kwargs):
        """Send multiplicative triples of given sizes to the parties"""
        a = generate_random_ring_element(size0)
        b = generate_random_ring_element(size1)
        c = getattr(torch, op)(a, b, *args, **kwargs)

        import crypten

        a = crypten.ArithmeticSharedTensor(a, precision=0, src=0)
        b = crypten.ArithmeticSharedTensor(b, precision=0, src=0)
        c = crypten.ArithmeticSharedTensor(c, precision=0, src=0)

        return a, b, c

    @staticmethod
    def square(size):
        """Send square double of given size to the parties"""
        r = generate_random_ring_element(size)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        import crypten

        stacked = torch.stack([r, r2])
        stacked = crypten.ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    @staticmethod
    def generate_xor_triple(size):
        """Send xor triples of given size to the parties"""
        a = generate_kbit_random_tensor(size)
        b = generate_kbit_random_tensor(size)
        c = a & b

        # Stack to vectorize scatter function
        import crypten

        abc = torch.stack([a, b, c])
        abc = crypten.BinarySharedTensor(abc, src=0)
        return abc[0], abc[1], abc[2]

    @staticmethod
    def wrap_rng(size, num_parties):
        """Send random shared tensor of given size and its wraps to the parties"""
        r = generate_random_ring_element(size)
        r = share(r, num_parties=num_parties)

        if num_parties == 1:
            r = [r]

        theta_r = count_wraps(r)

        import crypten

        r = crypten.ArithmeticSharedTensor.from_shares(r, src=0)
        theta_r = crypten.ArithmeticSharedTensor(theta_r, precision=0, src=0)

        return r, theta_r

    @staticmethod
    def B2A_rng(size):
        """Send random bit tensor as SPDZ and GMW to the parties"""
        # generate random bit
        r = generate_kbit_random_tensor(size, bitlength=1)

        import crypten

        rA = crypten.ArithmeticSharedTensor(r, precision=0, src=0)
        rB = crypten.BinarySharedTensor(r, src=0)

        return rA, rB

    @staticmethod
    def rand(*sizes):
        """Generate random ArithmeticSharedTensor uniform on [0, 1] to the parties"""
        samples = torch.rand(*sizes)

        import crypten

        return crypten.ArithmeticSharedTensor(samples, src=0)

    @staticmethod
    def bernoulli(tensor):
        """Generate random ArithmeticSharedTensor bernoulli on {0, 1} to the parties"""
        samples = torch.bernoulli(tensor)

        import crypten

        return crypten.ArithmeticSharedTensor(samples, src=0)

    @staticmethod
    def randperm(tensor_size):
        """
        Generate `tensor_size[:-1]` random ArithmeticSharedTensor permutations of
        the first `tensor_size[-1]` whole numbers
        """
        import crypten

        tensor_len = tensor_size[-1]
        nperms = int(torch.tensor(tensor_size[:-1]).prod().item())
        random_permutation = torch.stack(
            [torch.randperm(tensor_len) + 1 for _ in range(nperms)]
        ).view(tensor_size)
        return crypten.ArithmeticSharedTensor(random_permutation, precision=0, src=0)
