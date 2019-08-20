#!/usr/bin/env python3

import crypten.common.constants as constants
from crypten import comm


class Beaver:
    """
    Beaver class contains all protocols that make calls to the TTP to
    perform variable hiding
    """

    @staticmethod
    def AND(x, y, bits=constants.K):
        """Performs Beaver AND protocol for BinarySharedTensor tensors x and y"""
        import crypten

        a, b, c = crypten.TrustedThirdParty.generate_xor_triple(
            x.size(), bitlength=bits
        )

        # Stack to vectorize reveal
        eps_del = crypten.BinarySharedTensor.stack([x ^ a, y ^ b]).reveal()
        epsilon = eps_del[0]
        delta = eps_del[1]

        c._tensor ^= (epsilon & b._tensor) ^ (a._tensor & delta)
        if c._rank == 0:
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
        import crypten

        if comm.get_world_size() < 2:
            return crypten.ArithmeticSharedTensor(xB._tensor, src=0)

        rA, rB = crypten.TrustedThirdParty.B2A_rng(xB.size())

        z = (xB ^ rB).reveal()
        rA._tensor = rA._tensor * (1 - z) - rA._tensor * z
        if rA._rank == 0:
            rA._tensor += z
        return rA
