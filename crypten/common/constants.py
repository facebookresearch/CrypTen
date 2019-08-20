#!/usr/bin/env python3

# constants for encoding:
# Q = 2 ** 89 - 1 # 10th Mersenne prime
Q = 2 ** 63 - 25  # largest prime that fits in LongTensor
# Q = 2 ** 31 - 1  # largest prime for signed 32-bit integer
# Q = int(2 ** 64 - 1)  # 64 - bit integer
K = int(64)
LOG_K = int(6)

BASE = int(2)
PRECISION = int(16)


LONG_SIZE = 8
VERBOSE = True
