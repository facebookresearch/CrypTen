#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


# Cached SPK masks are:
# [0] -> 010101010101....0101  =                       01 x 32
# [1] -> 001000100010....0010  =                     0010 x 16
# [2] -> 000010000000....0010  =                 00001000 x  8
# [n] -> [2^n 0s, 1, (2^n -1) 0s] x (32 / (2^n))
__MASKS = torch.LongTensor(
    [
        6148914691236517205,
        2459565876494606882,
        578721382704613384,
        36029346783166592,
        140737488388096,
        2147483648,
    ]
)


def __fan(mask, iter):
    """Fans out bitmask from input to output at `iter` stage of the tree

    See arrows in Fig. 1 (right) of
    Catrina, O. "Improved Primitives for Secure Multiparty Integer Computation"
    """
    multiplier = (1 << (2 ** iter + 1)) - 2
    if isinstance(mask, (int, float)) or torch.is_tensor(mask):
        return mask * multiplier

    # Otherwise assume BinarySharedTensor
    result = mask.clone()
    result._tensor *= multiplier
    return result


def __SPK_circuit(S, P):
    """
    Computes the Set-Propagate-Kill Tree circuit for a set (S, P)
    (K is implied by S, P since (SPK) is one-hot)

    (See section 6.3 of Damgard, "Unconditionally Secure Constant-Rounds
    Multi-Party Computation for Equality, Comparison, Bits and Exponentiation")

    At each stage:
        S <- S0 ^ (P0 & S1)
        P <- P0 & P1
        K <- K0 ^ (P0 & K1) <- don't need K since it is implied by S and P
    """
    from .binary import BinarySharedTensor

    log_bits = int(math.log2(torch.iinfo(torch.long).bits))
    for i in range(log_bits):
        in_mask = __MASKS[i]
        out_mask = __fan(in_mask, i)
        not_out_mask = out_mask ^ -1

        # Set up S0, S1, P0, and P1
        P0 = P & out_mask
        P1 = __fan(P & in_mask, i)
        S1 = __fan(S & in_mask, i)

        # Vectorize private AND calls to reduce rounds:
        P0P0 = BinarySharedTensor.stack([P0, P0])
        S1P1 = BinarySharedTensor.stack([S1, P1])
        update = P0P0 & S1P1

        # Update S and P
        S ^= update[0]  # S ^= (P0 & S1)
        P = (P & not_out_mask) ^ update[1]  # P1 = P0 & P1
    return S, P


def add(x, y):
    """Returns x + y from BinarySharedTensors `x` and `y`"""
    S = x & y
    P = x ^ y
    carry, _ = __SPK_circuit(S, P)
    return P ^ (carry << 1)
