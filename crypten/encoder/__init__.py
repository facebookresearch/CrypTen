#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .fixed_point_encoder import FixedPointEncoder


__all__ = ["FixedPointEncoder"]

__SUPPORTED_ENCODERS = [FixedPointEncoder]
__default_encoder = __SUPPORTED_ENCODERS[0]


def get_default_encoder():
    return __default_encoder


def set_default_encoder(encoder):
    assert (
        encoder in __SUPPORTED_ENCODERS
    ), f"Provided encoder is not supported {encoder}"
    global __default_encoder
    __default_encoder = encoder


def set_default_precision(precision_bits):
    assert (
        isinstance(precision_bits, int) and precision_bits >= 0 and precision_bits < 64
    ), "precision must be a positive integer less than 64"
    FixedPointEncoder.set_default_precision(precision_bits)
