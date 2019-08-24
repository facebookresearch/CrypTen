#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

BITS = int(64)  # Number of bits used in each share
LOG_BITS = int(6)  # Log of the number of bits used in each share
PRECISION = int(16)  # Bits of precision used in fixed-point encoding
VERBOSE = True  # Determines whether communicator logs stats
