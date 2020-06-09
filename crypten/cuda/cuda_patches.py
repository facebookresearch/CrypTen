#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def __encode_as_fp64(x):
    """Converts a torch.cuda.LongTensor `x` to an encoding of
    torch.cuda.DoubleTensor that represent the same data.
    """

    x_block = torch.stack([(x >> (16 * i)) & (2 ** 16 - 1) for i in range(4)])

    return x_block.double()


def __decode_as_int64(x_enc):
    """converts an encoded torch.cuda.DoubleTensor `x` back to a
    the torch.cuda.LongTensor it encodes
    """
    x_enc = x_enc.long()

    x = (x_enc[3] + x_enc[6] + x_enc[9] + x_enc[12]) << 48
    x += (x_enc[2] + x_enc[5] + x_enc[8]) << 32
    x += (x_enc[1] + x_enc[4]) << 16
    x += x_enc[0]

    return x


def mul(x, y, *args, **kwargs):
    x_encoded = __encode_as_fp64(x)
    y_encoded = __encode_as_fp64(y)

    # span x and y for cross multiplication
    repeat_idx = [1] * (x_encoded.dim() - 1)
    x_enc_span = x_encoded.repeat(4, *repeat_idx)
    y_enc_span = y_encoded.repeat_interleave(repeats=4, dim=0)

    # expand the dimension of y to x so that broadcasting works as expected
    assert x_enc_span.ndim >= y_enc_span.ndim
    for _ in range(abs(x_enc_span.ndim - y_enc_span.ndim)):
        if x_enc_span.ndim > y_enc_span.ndim:
            y_enc_span.unsqueeze_(1)
        else:
            x_enc_span.unsqueeze_(1)

    z_encoded = torch.mul(x_enc_span, y_enc_span, *args, **kwargs)

    return __decode_as_int64(z_encoded)


def matmul(x, y, *args, **kwargs):
    x_encoded = __encode_as_fp64(x)
    y_encoded = __encode_as_fp64(y)

    # span x and y for cross multiplication
    repeat_idx = [1] * (x_encoded.dim() - 1)
    x_enc_span = x_encoded.repeat(4, *repeat_idx)
    y_enc_span = y_encoded.repeat_interleave(repeats=4, dim=0)

    z_encoded = torch.matmul(x_enc_span, y_enc_span, *args, **kwargs)

    return __decode_as_int64(z_encoded)
