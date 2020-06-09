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


def __patched_conv(op, x, y, *args, **kwargs):
    x_encoded = __encode_as_fp64(x)
    y_encoded = __encode_as_fp64(y)

    repeat_idx = [1] * (x_encoded.dim() - 1)
    x_enc_span = x_encoded.repeat(4, *repeat_idx)
    y_enc_span = y_encoded.repeat_interleave(repeats=4, dim=0)

    bs, c, *img = x.shape
    c_out, c_in, *ks = y.shape

    x_enc_span = x_enc_span.transpose_(0, 1).reshape(bs, 16 * c, *img)
    y_enc_span = y_enc_span.reshape(16 * c_out, c_in, *ks)

    c_z = c_out if op in ["conv1d", "conv2d"] else c_in

    z_encoded = getattr(torch, op)(x_enc_span, y_enc_span, *args, **kwargs, groups=16)
    z_encoded = z_encoded.reshape(bs, 16, c_z, *z_encoded.shape[2:]).transpose_(0, 1)

    return __decode_as_int64(z_encoded)


def __add_patched_operation(op):
    """
    Adds function to `MPCTensor` that is applied directly on the underlying
    `_tensor` attribute, and stores the result in the same attribute.
    """

    def patched_func(x, y, *args, **kwargs):
        return __patched_conv(op, x, y, *args, **kwargs)

    globals()[op] = patched_func


__patched_ops = ["conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d"]
for op in __patched_ops:
    __add_patched_operation(op)
