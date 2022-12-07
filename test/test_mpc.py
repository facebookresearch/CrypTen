#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import math
import os
import unittest

import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.functions.pooling import _pool2d_reshape
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.mpc import MPCTensor, ptype as Ptype
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestMPC(object):
    """
    This class tests all functions of MPCTensor.
    """

    def _get_random_test_tensor(self, *args, **kwargs):
        return get_random_test_tensor(device=self.device, *args, **kwargs)

    def _check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text(dst=dst)
        if dst is not None and dst != self.rank:
            self.assertIsNone(tensor)
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")

        if tensor.device != reference.device:
            tensor = tensor.cpu()
            reference = reference.cpu()

        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.2)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Reference %s" % reference)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def _check_tuple(self, encrypted_tuple, reference, msg, tolerance=None):
        self.assertTrue(isinstance(encrypted_tuple, tuple))
        self.assertEqual(len(encrypted_tuple), len(reference))
        for i in range(len(reference)):
            self._check(encrypted_tuple[i], reference[i], msg, tolerance=tolerance)

    def test_repr(self):
        a = self._get_random_test_tensor(size=(1,))
        arithmetic = MPCTensor(a, ptype=Ptype.arithmetic)
        binary = MPCTensor(a, ptype=Ptype.binary)

        # Make sure these don't crash
        print(arithmetic)
        repr(arithmetic)

        print(binary)
        repr(binary)

    def test_from_shares(self):
        """Tests MPCTensor.from_shares() functionality."""

        # settings for test:
        num_parties = int(self.world_size)
        size = (5, 4)

        def _generate_tensor(ptype):
            reference = self._get_random_test_tensor(size=size, is_float=False)

            # generate arithmetic sharing of reference tensor:
            if ptype == Ptype.arithmetic:
                zero_shares = generate_random_ring_element(
                    (num_parties, *size), device=self.device
                )
                zero_shares = zero_shares - zero_shares.roll(1, dims=0)
                shares = list(zero_shares.unbind(0))
                shares[0] += reference

            # generate binary sharing of reference tensor:
            else:
                zero_shares = generate_kbit_random_tensor(
                    (num_parties, *size), device=self.device
                )
                zero_shares = zero_shares ^ zero_shares.roll(1, dims=0)
                shares = list(zero_shares.unbind(0))
                shares[0] ^= reference

            # return shares and reference:
            return shares, reference

        # test both types:
        for ptype in [Ptype.arithmetic, Ptype.binary]:

            # generate shares, sync them between parties, and create tensor:
            shares, reference = _generate_tensor(ptype)
            share = comm.get().scatter(shares, 0)
            encrypted_tensor = MPCTensor.from_shares(share, ptype=ptype)

            # check resulting tensor:
            self.assertIsInstance(encrypted_tensor, MPCTensor)
            self.assertEqual(encrypted_tensor.ptype, ptype)
            self.assertIsInstance(encrypted_tensor._tensor, ptype.to_tensor())
            decrypted_tensor = encrypted_tensor.reveal()

            self.assertTrue(torch.all(decrypted_tensor.eq(reference)).item())

    def test_share_attr(self):
        """Tests share attribute getter and setter"""
        for is_float in (True, False):
            reference = self._get_random_test_tensor(is_float=is_float)
            encrypted_tensor = MPCTensor(reference)
            underlying_tensor = encrypted_tensor.share
            self.assertTrue(
                torch.equal(encrypted_tensor.share, underlying_tensor),
                "share getter failed",
            )

            new_share = self._get_random_test_tensor(is_float=False)
            encrypted_tensor.share = new_share
            self.assertTrue(
                torch.equal(encrypted_tensor.share, new_share), "share setter failed"
            )

    def test_encrypt_decrypt(self):
        """
        Tests tensor encryption and decryption for both positive
        and negative values.
        """
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            # encryption and decryption without source:
            reference = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(reference)
            self._check(encrypted_tensor, reference, "en/decryption failed")
            for dst in range(self.world_size):
                self._check(
                    encrypted_tensor, reference, "en/decryption failed", dst=dst
                )

            # test creation via new() function:
            encrypted_tensor2 = encrypted_tensor.new(reference)
            self.assertIsInstance(
                encrypted_tensor2, MPCTensor, "new() returns incorrect type"
            )
            self._check(encrypted_tensor2, reference, "en/decryption failed")

            # TODO: Implement broadcast_size on GPU
            if self.device.type == "cuda":
                continue

            # encryption and decryption with source:
            for src in range(self.world_size):
                input_tensor = reference if src == self.rank else []
                encrypted_tensor = MPCTensor(input_tensor, src=src, broadcast_size=True)
                for dst in range(self.world_size):
                    self._check(
                        encrypted_tensor,
                        reference,
                        "en/decryption with broadcast_size failed",
                        dst=dst,
                    )

        # MPCTensors cannot be initialized with None:
        with self.assertRaises(ValueError):
            _ = MPCTensor(None)

    def test_arithmetic(self):
        """Tests arithmetic functions on encrypted tensor."""
        arithmetic_functions = ["add", "add_", "sub", "sub_", "mul", "mul_"]
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True)
                encrypted = MPCTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)
                self._check(
                    encrypted_out,
                    reference,
                    "%s %s failed"
                    % ("private" if tensor_type == MPCTensor else "public", func),
                )
                if "_" in func:
                    # Check in-place op worked
                    self._check(
                        encrypted,
                        reference,
                        "%s %s failed"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )
                else:
                    # Check original is not modified
                    self._check(
                        encrypted,
                        tensor1,
                        "%s %s failed"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )

                # Check encrypted vector with encrypted scalar works.
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True, size=(1,))
                encrypted1 = MPCTensor(tensor1)
                encrypted2 = MPCTensor(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted1, func)(encrypted2)
                self._check(encrypted_out, reference, "private %s failed" % func)

            tensor = self._get_random_test_tensor(is_float=True)
            reference = tensor * tensor
            encrypted = MPCTensor(tensor)
            encrypted_out = encrypted.square()
            self._check(encrypted_out, reference, "square failed")

        # Test radd, rsub, and rmul
        reference = 2 + tensor1
        encrypted = MPCTensor(tensor1)
        encrypted_out = 2 + encrypted
        self._check(encrypted_out, reference, "right add failed")

        reference = 2 - tensor1
        encrypted_out = 2 - encrypted
        self._check(encrypted_out, reference, "right sub failed")

        reference = 2 * tensor1
        encrypted_out = 2 * encrypted
        self._check(encrypted_out, reference, "right mul failed")

    def test_sum(self):
        """Tests sum reduction on encrypted tensor."""
        tensor = self._get_random_test_tensor(size=(100, 100), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.sum(), tensor.sum(), "sum failed")

        for dim in [0, 1]:
            reference = tensor.sum(dim)
            encrypted_out = encrypted.sum(dim)
            self._check(encrypted_out, reference, "sum failed")

    def test_prod(self):
        """Tests prod reduction on encrypted tensor."""
        tensor = self._get_random_test_tensor(size=(3, 3), max_value=3, is_float=False)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.prod(), tensor.prod().float(), "prod failed")

        tensor = self._get_random_test_tensor(
            size=(5, 5, 5), max_value=3, is_float=False
        )
        encrypted = MPCTensor(tensor)
        for dim in [0, 1, 2]:
            reference = tensor.prod(dim).float()
            encrypted_out = encrypted.prod(dim)
            self._check(encrypted_out, reference, "prod failed")

    def test_ptype(self):
        """Test that ptype attribute creates the correct type of encrypted tensor"""
        ptype_values = [crypten.mpc.arithmetic, crypten.mpc.binary]
        tensor_types = [ArithmeticSharedTensor, BinarySharedTensor]
        for i, curr_ptype in enumerate(ptype_values):
            tensor = self._get_random_test_tensor(is_float=False)
            encr_tensor = crypten.cryptensor(tensor, ptype=curr_ptype)
            assert isinstance(encr_tensor._tensor, tensor_types[i]), "ptype test failed"

    def test_div(self):
        """Tests division of encrypted tensor by scalar and tensor."""
        for function in ["div", "div_"]:
            for scalar in [2, 2.0]:
                tensor = self._get_random_test_tensor(is_float=True)

                reference = tensor.float().div(scalar)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(scalar)
                self._check(encrypted_tensor, reference, "scalar division failed")

                # multiply denominator by 10 to avoid dividing by small num
                divisor = self._get_random_test_tensor(is_float=True, ex_zero=True) * 10
                reference = tensor.div(divisor)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(divisor)
                self._check(encrypted_tensor, reference, "tensor division failed")

    def test_mean(self):
        """Tests computing means of encrypted tensors."""
        tensor = self._get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.mean(), tensor.mean(), "mean failed")

        for dim in [0, 1, 2]:
            reference = tensor.mean(dim)
            encrypted_out = encrypted.mean(dim)
            self._check(encrypted_out, reference, "mean failed")

    def test_var(self):
        """Tests computing variances of encrypted tensors."""
        tensor = self._get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.var(), tensor.var(), "var failed")

        for dim in [0, 1, 2]:
            reference = tensor.var(dim)
            encrypted_out = encrypted.var(dim)
            self._check(encrypted_out, reference, "var failed")

    def test_matmul(self):
        """Test matrix multiplication."""
        for tensor_type in [lambda x: x, MPCTensor]:
            tensor = self._get_random_test_tensor(max_value=7, is_float=True)
            for width in range(2, tensor.nelement()):
                matrix_size = (tensor.nelement(), width)
                matrix = self._get_random_test_tensor(
                    max_value=7, size=matrix_size, is_float=True
                )
                reference = tensor.matmul(matrix)
                encrypted_tensor = MPCTensor(tensor)
                matrix = tensor_type(matrix)
                encrypted_tensor = encrypted_tensor.matmul(matrix)

                self._check(
                    encrypted_tensor,
                    reference,
                    "Private-%s matrix multiplication failed"
                    % ("private" if tensor_type == MPCTensor else "public"),
                )

    def test_dot_ger(self):
        """Test dot product of vector and encrypted tensor."""
        for tensor_type in [lambda x: x, MPCTensor]:
            tensor1 = self._get_random_test_tensor(is_float=True).squeeze()
            tensor2 = self._get_random_test_tensor(is_float=True).squeeze()
            dot_reference = tensor1.dot(tensor2)
            ger_reference = torch.ger(tensor1, tensor2)

            tensor2 = tensor_type(tensor2)

            # dot
            encrypted_tensor = MPCTensor(tensor1)
            encrypted_out = encrypted_tensor.dot(tensor2)
            self._check(
                encrypted_out,
                dot_reference,
                "%s dot product failed" % "private"
                if tensor_type == MPCTensor
                else "public",
            )

            # ger
            encrypted_tensor = MPCTensor(tensor1)
            encrypted_out = encrypted_tensor.ger(tensor2)
            self._check(
                encrypted_out,
                ger_reference,
                "%s outer product failed" % "private"
                if tensor_type == MPCTensor
                else "public",
            )

    def test_squeeze(self):
        tensor = self._get_random_test_tensor(is_float=True)
        for dim in [0, 1, 2]:
            # Test unsqueeze
            reference = tensor.unsqueeze(dim)

            encrypted = MPCTensor(tensor)
            encrypted_out = encrypted.unsqueeze(dim)
            self._check(encrypted_out, reference, "unsqueeze failed")

            # Test squeeze
            encrypted = MPCTensor(tensor.unsqueeze(0))
            encrypted_out = encrypted.squeeze()
            self._check(encrypted_out, reference.squeeze(), "squeeze failed")

            # Check that the encrypted_out and encrypted point to the same
            # thing.
            encrypted_out[0:2] = torch.tensor(
                [0, 1], dtype=torch.float, device=self.device
            )
            ref = encrypted.squeeze().get_plain_text()
            self._check(encrypted_out, ref, "squeeze failed")

    def test_transpose(self):
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            if len(size) == 2:  # t() asserts dim == 2
                reference = tensor.t()
                encrypted_out = encrypted_tensor.t()
                self._check(encrypted_out, reference, "t() failed")

            for dim0 in range(len(size)):
                for dim1 in range(len(size)):
                    reference = tensor.transpose(dim0, dim1)
                    encrypted_out = encrypted_tensor.transpose(dim0, dim1)
                    self._check(encrypted_out, reference, "transpose failed")

    def test_conv1d_smaller_signal_one_channel(self):
        self._conv1d(5, 1)

    def test_conv1d_smaller_signal_many_channels(self):
        self._conv1d(5, 5)

    def test_conv1d_larger_signal_one_channel(self):
        self._conv1d(16, 1)

    def test_conv1d_larger_signal_many_channels(self):
        self._conv1d(16, 5)

    def _conv1d(self, signal_size, in_channels):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        kernel_sizes = [1, 2, 3]
        ochannels = [1, 3, 6]
        paddings = [0, 1]
        strides = [1, 2]
        dilations = [1, 2]
        groupings = [1, 2]

        for func_name in ["conv1d", "conv_transpose1d"]:
            for kernel_type in [lambda x: x, MPCTensor]:
                for (
                    batches,
                    kernel_size,
                    out_channels,
                    padding,
                    stride,
                    dilation,
                    groups,
                ) in itertools.product(
                    nbatches,
                    kernel_sizes,
                    ochannels,
                    paddings,
                    strides,
                    dilations,
                    groupings,
                ):
                    # group convolution is not supported on GPU
                    if self.device.type == "cuda" and groups > 1:
                        continue

                    input_size = (batches, in_channels * groups, signal_size)
                    signal = self._get_random_test_tensor(
                        size=input_size, is_float=True
                    )

                    if func_name == "conv1d":
                        k_size = (out_channels * groups, in_channels, kernel_size)
                    else:
                        k_size = (in_channels * groups, out_channels, kernel_size)
                    kernel = self._get_random_test_tensor(size=k_size, is_float=True)

                    reference = getattr(F, func_name)(
                        signal,
                        kernel,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                        groups=groups,
                    )
                    encrypted_signal = MPCTensor(signal)
                    encrypted_kernel = kernel_type(kernel)
                    encrypted_conv = getattr(encrypted_signal, func_name)(
                        encrypted_kernel,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                        groups=groups,
                    )

                    self._check(encrypted_conv, reference, f"{func_name} failed")

    def test_conv2d_square_image_one_channel(self):
        self._conv2d((5, 5), 1, "conv2d")

    def test_conv_transpose2d_square_image_one_channel(self):
        self._conv2d((5, 5), 1, "conv_transpose2d")

    def test_conv2d_square_image_many_channels(self):
        self._conv2d((5, 5), 5, "conv2d")

    def test_conv_transpose2d_square_image_many_channels(self):
        self._conv2d((5, 5), 5, "conv_transpose2d")

    def test_conv2d_rectangular_image_one_channel(self):
        self._conv2d((16, 7), 1, "conv2d")

    def test_conv_transpose2d_rectangular_image_one_channel(self):
        self._conv2d((16, 7), 1, "conv_transpose2d")

    def test_conv2d_rectangular_image_many_channels(self):
        self._conv2d((16, 7), 5, "conv2d")

    def test_conv_transpose2d_rectangular_image_many_channels(self):
        self._conv2d((16, 7), 5, "conv_transpose2d")

    def _conv2d(self, image_size, in_channels, func_name):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        ochannels = [1, 3]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]
        dilations = [1, 2]
        groupings = [1, 2]

        assert func_name in [
            "conv2d",
            "conv_transpose2d",
        ], f"Invalid func_name: {func_name}"

        for kernel_type in [lambda x: x, MPCTensor]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
                dilation,
                groups,
            ) in itertools.product(
                nbatches,
                kernel_sizes,
                ochannels,
                paddings,
                strides,
                dilations,
                groupings,
            ):
                # group convolution is not supported on GPU
                if self.device.type == "cuda" and groups > 1:
                    continue

                # sample input:
                input_size = (batches, in_channels * groups, *image_size)
                input = self._get_random_test_tensor(size=input_size, is_float=True)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels * groups, in_channels, *kernel_size)
                else:
                    k_size = (in_channels * groups, out_channels, *kernel_size)
                kernel = self._get_random_test_tensor(size=k_size, is_float=True)

                # perform filtering:
                encr_matrix = MPCTensor(input)
                encr_kernel = kernel_type(kernel)
                encr_conv = getattr(encr_matrix, func_name)(
                    encr_kernel,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    groups=groups,
                )

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input,
                    kernel,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    groups=groups,
                )
                self._check(encr_conv, reference, "%s failed" % func_name)

    def test_max_pooling(self):
        """Test max_pool of encrypted tensor."""

        def _assert_index_match(
            indices,
            encrypted_indices,
            matrix_size,
            kernel_size,
            **kwargs,
        ):
            # Assert each kernel is one-hot
            self.assertTrue(
                encrypted_indices.get_plain_text()
                .sum(-1)
                .sum(-1)
                .eq(torch.ones_like(indices))
                .all(),
                "Encrypted indices are not one-hot",
            )

            # Populate tensor with kernel indices
            arange_size = matrix_size[-2:]
            index_values = torch.arange(arange_size.numel(), device=indices.device)
            index_values = index_values.view(arange_size)
            index_values = index_values.expand(matrix_size)

            # Ensure encrypted indices are correct
            index_mask, size = _pool2d_reshape(index_values, kernel_size, **kwargs)
            index_mask = index_mask.view(*size, kernel_size, kernel_size)
            crypten_indices = encrypted_indices.mul(index_mask).sum(-1).sum(-1)

            self._check(
                crypten_indices, indices.float(), "max_pool2d indexing is incorrect"
            )

        dilations = [1, 2]
        for width in range(2, 5):
            for kernel_size in range(1, width):
                matrix_size = (1, 4, 5, width)
                matrix = self._get_random_test_tensor(size=matrix_size, is_float=True)

                strides = list(range(1, kernel_size + 1)) + [(1, kernel_size)]
                paddings = range(kernel_size // 2 + 1)

                for (
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    return_indices,
                ) in itertools.product(
                    strides,
                    paddings,
                    dilations,
                    [False, True],
                    [False, True],
                ):
                    kwargs = {
                        "stride": stride,
                        "padding": padding,
                        "dilation": dilation,
                        "ceil_mode": ceil_mode,
                        "return_indices": return_indices,
                    }

                    # Skip kernels that lead to 0-size outputs
                    if (kernel_size - 1) * dilation > width - 1:
                        continue

                    reference = F.max_pool2d(matrix, kernel_size, **kwargs)
                    encrypted_matrix = MPCTensor(matrix)
                    encrypted_pool = encrypted_matrix.max_pool2d(kernel_size, **kwargs)

                    if return_indices:
                        indices = reference[1]
                        encrypted_indices = encrypted_pool[1]

                        kwargs.pop("return_indices")
                        _assert_index_match(
                            indices,
                            encrypted_indices,
                            matrix.size(),
                            kernel_size,
                            **kwargs,
                        )

                        encrypted_pool = encrypted_pool[0]
                        reference = reference[0]

                    self._check(encrypted_pool, reference, "max_pool2d failed")

    def test_avg_pooling(self):
        """Test avg_pool of encrypted tensor."""
        for width in range(2, 5):
            for kernel_size in range(1, width):
                matrix_size = (1, 4, 5, width)
                matrix = self._get_random_test_tensor(size=matrix_size, is_float=True)

                strides = list(range(1, kernel_size + 1)) + [(1, kernel_size)]
                paddings = range(kernel_size // 2 + 1)

                for stride, padding in itertools.product(strides, paddings):
                    kwargs = {"stride": stride, "padding": padding}
                    reference = F.avg_pool2d(matrix, kernel_size, **kwargs)

                    encrypted_matrix = MPCTensor(matrix)
                    encrypted_pool = encrypted_matrix.avg_pool2d(kernel_size, **kwargs)
                    self._check(encrypted_pool, reference, "avg_pool2d failed")

    def test_adaptive_pooling(self):
        """test adaptive_avg_pool2d and adaptive_max_pool2d"""
        for in_size in range(1, 11):
            for out_size in list(range(1, in_size + 1)) + [None]:
                input_size = (1, in_size, in_size)
                output_size = (out_size, out_size)

                tensor = self._get_random_test_tensor(
                    size=input_size, is_float=True
                ).unsqueeze(0)
                encrypted = MPCTensor(tensor)

                # Test adaptive_avg_pool2d
                reference = F.adaptive_avg_pool2d(tensor, output_size)
                encrypted_out = encrypted.adaptive_avg_pool2d(output_size)
                self._check(encrypted_out, reference, "adaptive_avg_pool2d failed")

                # Test adapvite_max_pool2d
                for return_indices in [False, True]:
                    reference = F.adaptive_max_pool2d(
                        tensor, output_size, return_indices=return_indices
                    )
                    encrypted_out = encrypted.adaptive_max_pool2d(
                        output_size, return_indices=return_indices
                    )

                    if return_indices:
                        encrypted_out = encrypted_out[0]
                        reference = reference[0]
                    self._check(encrypted_out, reference, "adaptive_max_pool2d failed")

    def test_take(self):
        """Tests take function on encrypted tensor"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor(
            [[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long, device=self.device
        )
        tensor = self._get_random_test_tensor(size=tensor_size, is_float=True)

        # Test when dimension!=None
        for dimension in range(0, 4):
            ndarray = tensor.cpu().numpy()
            reference = torch.from_numpy(ndarray.take(index.cpu(), dimension))
            encrypted_tensor = MPCTensor(tensor)
            encrypted_out = encrypted_tensor.take(index, dimension)
            self._check(encrypted_out, reference, "take function failed: dimension set")

        # Test when dimension is default (i.e. None)
        sizes = [(15,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            take_indices = [[0], [10], [0, 5, 10]]
            for indices in take_indices:
                indices = torch.tensor(indices, device=self.device)
                self._check(
                    encrypted_tensor.take(indices),
                    tensor.take(indices),
                    f"take failed with indices {indices}",
                )

    def test_neg(self):
        """Test negative on encrypted tensor."""
        for width in range(2, 5):
            matrix_size = (5, width)
            matrix = self._get_random_test_tensor(size=matrix_size, is_float=True)
            encrypted_matrix = MPCTensor(matrix)
            self._check(-encrypted_matrix, -matrix, "__neg__ failed")
            for func_name in ["neg", "neg_"]:
                reference = getattr(matrix, func_name)()
                encrypted_output = getattr(encrypted_matrix, func_name)()
                self._check(encrypted_output, reference, "%s failed" % func_name)

    def test_relu(self):
        """Test relu on encrypted tensor."""
        for width in range(2, 5):
            matrix_size = (5, width)
            matrix = self._get_random_test_tensor(size=matrix_size, is_float=True)

            # Generate some negative values
            matrix2 = self._get_random_test_tensor(size=matrix_size, is_float=True)
            matrix = matrix - matrix2

            encrypted_matrix = MPCTensor(matrix)
            reference = F.relu_(matrix)
            encrypted_matrix = encrypted_matrix.relu()
            self._check(encrypted_matrix, reference, "relu failed")

    def test_comparators(self):
        """Test comparators (>, >=, <, <=, ==, !=)"""
        for comp in ["gt", "ge", "lt", "le", "eq", "ne"]:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True)

                encrypted_tensor1 = MPCTensor(tensor1)
                encrypted_tensor2 = tensor_type(tensor2)

                reference = getattr(tensor1, comp)(tensor2).float()
                encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)

                self._check(encrypted_out, reference, "%s comparator failed" % comp)

                # Check deterministic example to guarantee all combinations
                tensor1 = torch.tensor([2.0, 3.0, 1.0, 2.0, 2.0])
                tensor2 = torch.tensor([2.0, 2.0, 2.0, 3.0, 1.0])

                encrypted_tensor1 = MPCTensor(tensor1)
                encrypted_tensor2 = tensor_type(tensor2)

                reference = getattr(tensor1, comp)(tensor2).float()
                encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)

                self._check(encrypted_out, reference, "%s comparator failed" % comp)

    def test_max_min_pairwise(self):
        """Tests max and min for the deterministic constant (n^2) algorithm"""
        self._max_min("pairwise")

    def test_max_min_log_reduction(self):
        """Tests max and min for log reduction algorithm"""
        self._max_min("log_reduction")

    def test_max_min_double_log_reduction(self):
        """Tests max and min for double log reduction algorithm"""
        self._max_min("double_log_reduction")

    def test_max_min_accelerated_cascade(self):
        """Tests max and min for accelerated cascading algorithm"""
        self._max_min("accelerated_cascade")

    def _max_min(self, method):
        """Test max and min for the specified algorithm"""
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        test_cases = [
            torch.tensor(
                [[1, 1, 2, 1, 4, 1, 3, 4]], dtype=torch.float, device=self.device
            )
        ] + [self._get_random_test_tensor(size=size, is_float=False) for size in sizes]

        for tensor in test_cases:
            tensor = tensor.float()
            encrypted_tensor = MPCTensor(tensor)
            for comp in ["max", "min"]:
                reference = getattr(tensor, comp)()
                with cfg.temp_override({"functions.max_method": method}):
                    encrypted_out = getattr(encrypted_tensor, comp)()
                self._check(encrypted_out, reference, "%s reduction failed" % comp)

                for dim in range(tensor.dim()):
                    for keepdim in [False, True]:
                        reference = getattr(tensor, comp)(dim, keepdim=keepdim)

                        # Test with one_hot = False
                        with cfg.temp_override({"functions.max_method": method}):
                            encrypted_out = getattr(encrypted_tensor, comp)(
                                dim, keepdim=keepdim, one_hot=False
                            )

                        # Check max / min values are correct
                        self._check(
                            encrypted_out[0], reference[0], "%s reduction failed" % comp
                        )

                        # Test argmax / argmin values are correct
                        out_encr = encrypted_out[1]
                        out_decr = out_encr.get_plain_text().long()
                        argmax_ref = reference[1]

                        # Must index into tensor since ties are broken randomly
                        # so crypten and PyTorch can return different indices.
                        # This checks that they index to the same value.
                        if not keepdim:
                            out_decr = out_decr.unsqueeze(dim)
                            argmax_ref = argmax_ref.unsqueeze(dim)
                        mpc_result = tensor.gather(dim, out_decr)
                        torch_result = tensor.gather(dim, argmax_ref)
                        self.assertTrue(
                            (mpc_result == torch_result).all().item(),
                            "%s reduction failed" % comp,
                        )

                        # Test indices with one_hot = True
                        with cfg.temp_override({"functions.max_method": method}):
                            encrypted_out = getattr(encrypted_tensor, comp)(
                                dim, keepdim=keepdim, one_hot=True
                            )
                        # Check argmax results
                        val_ref = reference[0]
                        out_encr = encrypted_out[1]
                        out_decr = out_encr.get_plain_text()
                        self.assertTrue((out_decr.sum(dim) == 1).all())
                        self.assertTrue(
                            (
                                out_decr.mul(tensor).sum(dim, keepdim=keepdim)
                                == val_ref
                            ).all()
                        )

    def test_argmax_argmin_pairwise(self):
        """Tests argmax and argmin for the deterministic constant (n^2) algorithm"""
        self._argmax_argmin("pairwise")

    def test_argmax_argmin_log_reduction(self):
        """Tests argmax and argmin for log reduction algorithm"""
        self._argmax_argmin("log_reduction")

    def test_argmax_argmin_double_log_reduction(self):
        """Tests argmax and argmin for double log reduction algorithm"""
        self._argmax_argmin("double_log_reduction")

    def test_argmax_argmin_accelerated_cascade(self):
        """Tests max and min for accelerated cascading algorithm"""
        self._max_min("accelerated_cascade")

    def _argmax_argmin(self, method):
        """Test argmax and argmin for specified algorithm"""
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        test_cases = [
            torch.tensor(
                [[1, 1, 2, 1, 4, 1, 3, 4]], dtype=torch.float, device=self.device
            )
        ] + [self._get_random_test_tensor(size=size, is_float=False) for size in sizes]

        for tensor in test_cases:
            tensor = tensor.float()
            encrypted_tensor = MPCTensor(tensor)
            for comp in ["argmax", "argmin"]:
                cmp = comp[3:]

                value = getattr(tensor, cmp)()

                # test with one_hot = False
                with cfg.temp_override({"functions.max_method": method}):
                    encrypted_out = getattr(encrypted_tensor, comp)(one_hot=False)

                # Must index into tensor since ties are broken randomly
                # so crypten and PyTorch can return different indices.
                # This checks that they index to the same value.
                decrypted_out = encrypted_out.get_plain_text()
                if tensor.dim() == 0:  # if input is 0-d, argmax should be 0
                    self.assertEqual(decrypted_out, 0)
                else:
                    decrypted_val = tensor.flatten()[decrypted_out.long()]
                    self.assertTrue(decrypted_val.eq(value).all().item())

                # test with one_hot = False
                with cfg.temp_override({"functions.max_method": method}):
                    encrypted_out = getattr(encrypted_tensor, comp)(one_hot=True)
                one_hot_indices = (tensor == value).float()
                decrypted_out = encrypted_out.get_plain_text()
                self.assertTrue(decrypted_out.sum() == 1)
                self.assertTrue(decrypted_out.mul(one_hot_indices).sum() == 1)

                for dim in range(tensor.dim()):
                    for keepdim in [False, True]:
                        # Compute one-hot argmax/min reference in plaintext
                        values, indices = getattr(tensor, cmp)(dim, keepdim=keepdim)

                        # test with one_hot = False
                        with cfg.temp_override({"functions.max_method": method}):
                            encrypted_out = getattr(encrypted_tensor, comp)(
                                dim, keepdim=keepdim, one_hot=False
                            )

                        # Must index into tensor since ties are broken randomly
                        # so crypten and PyTorch can return different indices.
                        # This checks that they index to the same value.abs
                        decrypted_out = encrypted_out.get_plain_text()
                        if not keepdim:
                            decrypted_out = decrypted_out.unsqueeze(dim)
                            indices = indices.unsqueeze(dim)
                        decrypted_val = tensor.gather(dim, decrypted_out.long())
                        reference = tensor.gather(dim, indices)
                        self.assertTrue(decrypted_val.eq(reference).all().item())

                        # test with one_hot = True
                        with cfg.temp_override({"functions.max_method": method}):
                            encrypted_out = getattr(encrypted_tensor, comp)(
                                dim, keepdim=keepdim, one_hot=True
                            )
                        decrypted_out = encrypted_out.get_plain_text()

                        if not keepdim:
                            values = values.unsqueeze(dim)
                        one_hot_indices = tensor.eq(values).float()
                        self.assertTrue(decrypted_out.sum(dim).eq(1).all())
                        self.assertTrue(
                            decrypted_out.mul(one_hot_indices).sum(dim).eq(1).all()
                        )

    def test_abs_sign(self):
        """Test absolute value function"""
        for op in ["abs", "sign"]:
            tensor = self._get_random_test_tensor(is_float=True)
            if op == "sign":
                # do not test on 0 since torch.tensor([0]).sign() = 0
                tensor = tensor + (tensor == 0).float()
            encrypted_tensor = MPCTensor(tensor)
            reference = getattr(tensor, op)()

            encrypted_out = getattr(encrypted_tensor, op)()

            self._check(encrypted_out, reference, "%s failed" % op)

    def test_approximations(self):
        """Test appoximate functions (exp, log, sqrt, reciprocal, pos_pow)"""

        def test_with_inputs(func, input):
            encrypted_tensor = MPCTensor(input)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

        # Test on [-10, 10] range
        full_range_cases = ["exp"]
        tensor = torch.tensor(
            [0.01 * i for i in range(-1000, 1001, 1)], device=self.device
        )
        for func in full_range_cases:
            test_with_inputs(func, tensor)

        # Test on [0, 10] range
        tensor[tensor == 0] = 1.0
        non_zero_cases = ["reciprocal"]
        for func in non_zero_cases:
            test_with_inputs(func, tensor)

        # Test on [0, 10] range
        tensor = tensor[1001:]
        pos_cases = ["log", "sqrt"]
        for func in pos_cases:
            test_with_inputs(func, tensor)

        # Test pos_pow with several exponents
        encrypted_tensor = MPCTensor(tensor)

        # Reduced the max_value so approximations have less absolute error
        tensor_exponent = self._get_random_test_tensor(
            max_value=2, size=tensor.size(), is_float=True
        )
        exponents = [-3, -2, -1, 0, 1, 2, 3, tensor_exponent]
        exponents += [MPCTensor(tensor_exponent)]
        for p in exponents:
            if isinstance(p, MPCTensor):
                reference = tensor.pow(p.get_plain_text())
            else:
                reference = tensor.pow(p)
                encrypted_out = encrypted_tensor.pos_pow(p)
            self._check(encrypted_out, reference, f"pos_pow failed with power {p}")

    def test_norm(self):
        """Tests p-norm"""
        for p in [1, 1.5, 2, 3, float("inf"), "fro"]:
            for dim in [None, 0, 1, 2]:
                tensor = self._get_random_test_tensor(size=(3, 3, 3), is_float=True) / 5
                if dim is None:
                    reference = tensor.norm(p=p)
                else:
                    reference = tensor.norm(p=p, dim=dim)

                encrypted = MPCTensor(tensor)
                encrypted_out = encrypted.norm(p=p, dim=dim)
                self._check(encrypted_out, reference, f"{p}-norm failed", tolerance=0.5)

    def test_logistic(self):
        """Tests logistic functions (sigmoid, tanh)"""
        tensor = torch.tensor(
            [0.01 * i for i in range(-1000, 1001, 1)], device=self.device
        )
        encrypted_tensor = MPCTensor(tensor)

        cases = ["sigmoid", "tanh"]
        for func in cases:
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_hardtanh(self):
        tensor = torch.arange(-10, 10, dtype=torch.float32)
        encrypted = MPCTensor(tensor)

        for minval in range(-10, 10):
            for maxval in range(minval, 11):
                reference = torch.nn.functional.hardtanh(tensor, minval, maxval)
                encrypted_out = encrypted.hardtanh(minval, maxval)

                self._check(encrypted_out, reference, "hardtanh failed")

    def test_inplace_warning(self):
        """Tests that a warning is thrown that indicates that the `inplace` kwarg
        is ignored when a function is called with `inplace=True`
        """
        tensor = get_random_test_tensor(is_float=True)
        encrypted = MPCTensor(tensor)

        functions = ["dropout", "_feature_dropout"]
        for func in functions:
            warning_str = (
                f"CrypTen {func} does not support inplace computation during training."
            )
            with self.assertLogs(logger=logging.getLogger(), level="WARNING") as cm:
                getattr(encrypted, func)(inplace=True)
            self.assertTrue(f"WARNING:root:{warning_str}" in cm.output)

    def test_cos_sin(self):
        """Tests trigonometric functions (cos, sin)"""
        tensor = torch.tensor(
            [0.01 * i for i in range(-1000, 1001, 1)], device=self.device
        )
        encrypted_tensor = MPCTensor(tensor)

        cases = ["cos", "sin"]
        for func in cases:
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_softmax(self):
        """Test softmax and log_softmax function"""
        for softmax_fn in ["softmax", "log_softmax"]:
            # Test 0-dim tensor:
            tensor = self._get_random_test_tensor(size=(), is_float=True)
            reference = getattr(tensor, softmax_fn)(0)
            encrypted_tensor = MPCTensor(tensor)
            encrypted_out = getattr(encrypted_tensor, softmax_fn)(0)
            self._check(encrypted_out, reference, "0-dim tensor %s failed" % softmax_fn)

            # Test all other sizes
            sizes = [
                (1,),
                (5,),
                (1, 1),
                (1, 5),
                (5, 1),
                (5, 5),
                (1, 5, 5),
                (5, 1, 5),
                (5, 5, 1),
                (5, 5, 5),
                (1, 5, 5, 5),
                (5, 5, 5, 5),
            ]
            for size in sizes:
                tensor = self._get_random_test_tensor(size=size, is_float=True) / 5
                encrypted_tensor = MPCTensor(tensor)

                for dim in range(tensor.dim()):
                    reference = getattr(tensor, softmax_fn)(dim)
                    encrypted_out = getattr(encrypted_tensor, softmax_fn)(dim)

                    self._check(encrypted_out, reference, "%s failed" % softmax_fn)

    def test_get_set(self):
        """Tests element setting and getting by index"""
        for tensor_type in [lambda x: x, MPCTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = self._get_random_test_tensor(size=(size, size), is_float=True)
                reference = tensor[:, 0]

                encrypted_tensor = MPCTensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = self._get_random_test_tensor(size=(size,), is_float=True)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = MPCTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = MPCTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

    def test_pad(self):
        """Tests padding"""
        sizes = [(1,), (5,), (1, 1), (5, 5), (5, 5, 5), (5, 3, 32, 32)]
        pads = [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 1, 1),
            (2, 2, 1, 1),
            (2, 2, 2, 2),
        ]

        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            for pad in pads:
                for value in [0, 1, 10]:
                    if tensor.dim() < 2:
                        pad = pad[:2]
                    reference = torch.nn.functional.pad(tensor, pad, value=value)

                    encrypted_value = MPCTensor(value, device=self.device)
                    encrypted_out = encrypted_tensor.pad(pad, value=encrypted_value)
                    encrypted_out2 = encrypted_tensor.pad(pad, value=value)
                    self._check(encrypted_out, reference, "pad failed")
                    self._check(encrypted_out2, reference, "pad failed")

    def test_index_add(self):
        """Test index_add function of encrypted tensor"""
        index_add_functions = ["index_add", "index_add_"]
        tensor_size1 = [5, 5, 5, 5]
        index = torch.tensor(
            [1, 2, 3, 4, 4, 2, 1, 3], dtype=torch.long, device=self.device
        )
        for dimension in range(0, 4):
            tensor_size2 = [5, 5, 5, 5]
            tensor_size2[dimension] = index.size(0)
            for func in index_add_functions:
                for tensor_type in [lambda x: x, MPCTensor]:
                    tensor1 = self._get_random_test_tensor(
                        size=tensor_size1, is_float=True
                    )
                    tensor2 = self._get_random_test_tensor(
                        size=tensor_size2, is_float=True
                    )
                    encrypted = MPCTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(dimension, index, tensor2)
                    encrypted_out = getattr(encrypted, func)(
                        dimension, index, encrypted2
                    )
                    private_type = tensor_type == MPCTensor
                    self._check(
                        encrypted_out,
                        reference,
                        "%s %s failed"
                        % ("private" if private_type else "public", func),
                    )
                    if func.endswith("_"):
                        # Check in-place index_add worked
                        self._check(
                            encrypted,
                            reference,
                            "%s %s failed"
                            % ("private" if private_type else "public", func),
                        )
                    else:
                        # Check original is not modified
                        self._check(
                            encrypted,
                            tensor1,
                            "%s %s failed"
                            % (
                                "private" if tensor_type == MPCTensor else "public",
                                func,
                            ),
                        )

    def test_scatter(self):
        """Test scatter/scatter_add function of encrypted tensor"""
        funcs = ["scatter", "scatter_", "scatter_add", "scatter_add_"]
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for func in funcs:
            for size in sizes:
                for tensor_type in [lambda x: x, MPCTensor]:
                    for dim in range(len(size)):
                        tensor1 = self._get_random_test_tensor(size=size, is_float=True)
                        tensor2 = self._get_random_test_tensor(size=size, is_float=True)
                        index = self._get_random_test_tensor(size=size, is_float=False)
                        index = index.abs().clamp(0, 4)
                        encrypted = MPCTensor(tensor1)
                        encrypted2 = tensor_type(tensor2)
                        reference = getattr(tensor1, func)(dim, index, tensor2)
                        encrypted_out = getattr(encrypted, func)(dim, index, encrypted2)
                        private = tensor_type == MPCTensor
                        self._check(
                            encrypted_out,
                            reference,
                            "%s %s failed" % ("private" if private else "public", func),
                        )
                        if func.endswith("_"):
                            # Check in-place scatter/scatter_add worked
                            self._check(
                                encrypted,
                                reference,
                                "%s %s failed"
                                % ("private" if private else "public", func),
                            )
                        else:
                            # Check original is not modified
                            self._check(
                                encrypted,
                                tensor1,
                                "%s %s failed"
                                % ("private" if private else "public", func),
                            )

    def test_broadcast_arithmetic_ops(self):
        """Test broadcast of arithmetic functions."""
        arithmetic_functions = ["add", "sub", "mul", "div"]
        # TODO: Add broadcasting for pos_pow since it can take a tensor argument
        arithmetic_sizes = [
            (),
            (1,),
            (2,),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 2),
            (1, 1, 1, 1),
            (1, 1, 1, 2),
            (1, 1, 2, 1),
            (1, 2, 1, 1),
            (2, 1, 1, 1),
            (2, 2, 2, 2),
        ]

        for tensor_type in [lambda x: x, MPCTensor]:
            for func in arithmetic_functions:
                for size1, size2 in itertools.combinations(arithmetic_sizes, 2):
                    exclude_zero = True if func == "div" else False
                    # multiply denominator by 10 to avoid dividing by small num
                    const = 10 if func == "div" else 1

                    tensor1 = self._get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = self._get_random_test_tensor(
                        size=size2, is_float=True, ex_zero=exclude_zero
                    )
                    tensor2 *= const
                    encrypted1 = MPCTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(tensor2)
                    encrypted_out = getattr(encrypted1, func)(encrypted2)

                    private = isinstance(encrypted2, MPCTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s %s broadcast failed"
                        % ("private" if private else "public", func),
                    )

                    # Test with integer tensor
                    tensor2 = self._get_random_test_tensor(
                        size=size2, is_float=False, ex_zero=exclude_zero
                    )
                    tensor2 *= const
                    reference = getattr(tensor1, func)(tensor2.float())
                    encrypted_out = getattr(encrypted1, func)(tensor2)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s broadcast failed with public integer tensor" % func,
                    )

    def test_broadcast_matmul(self):
        """Test broadcast of matmul."""
        matmul_sizes = [(1, 1), (1, 5), (5, 1), (5, 5)]
        batch_dims = [(), (1,), (5,), (1, 1), (1, 5), (5, 5)]

        for tensor_type in [lambda x: x, MPCTensor]:

            for size in matmul_sizes:
                for batch1, batch2 in itertools.combinations(batch_dims, 2):
                    size1 = (*batch1, *size)
                    size2 = (*batch2, *size)

                    tensor1 = self._get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = self._get_random_test_tensor(size=size2, is_float=True)
                    tensor2 = tensor2.transpose(-2, -1)

                    encrypted1 = MPCTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)

                    reference = tensor1.matmul(tensor2)
                    encrypted_out = encrypted1.matmul(encrypted2)
                    private = isinstance(encrypted2, MPCTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s matmul broadcast failed"
                        % ("private" if private else "public"),
                    )

                    # Test with integer tensor
                    tensor2 = self._get_random_test_tensor(size=size2, is_float=False)
                    tensor2 = tensor2.float().transpose(-2, -1)
                    reference = tensor1.matmul(tensor2)
                    encrypted_out = encrypted1.matmul(tensor2)
                    self._check(
                        encrypted_out,
                        reference,
                        "matmul broadcast failed with public integer tensor",
                    )

    def test_inplace(self):
        """Test inplace vs. out-of-place functions"""
        for op in ["add", "sub", "mul", "div"]:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True)

                reference = getattr(torch, op)(tensor1, tensor2)

                encrypted1 = MPCTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                input_tensor_id = id(encrypted1._tensor)
                input_encrypted_id = id(encrypted1)

                # Test that out-of-place functions do not modify the input
                private = isinstance(encrypted2, MPCTensor)
                encrypted_out = getattr(encrypted1, op)(encrypted2)
                self._check(
                    encrypted1,
                    tensor1,
                    "%s out-of-place %s modifies input"
                    % ("private" if private else "public", op),
                )
                self._check(
                    encrypted_out,
                    reference,
                    "%s out-of-place %s produces incorrect output"
                    % ("private" if private else "public", op),
                )
                self.assertFalse(id(encrypted_out._tensor) == input_tensor_id)
                self.assertFalse(id(encrypted_out) == input_encrypted_id)

                # Test that in-place functions modify the input
                encrypted_out = getattr(encrypted1, op + "_")(encrypted2)
                self._check(
                    encrypted1,
                    reference,
                    "%s in-place %s_ does not modify input"
                    % ("private" if private else "public", op),
                )
                self._check(
                    encrypted_out,
                    reference,
                    "%s in-place %s_ produces incorrect output"
                    % ("private" if private else "public", op),
                )
                self.assertTrue(id(encrypted_out._tensor) == input_tensor_id)
                self.assertTrue(id(encrypted_out) == input_encrypted_id)

    def test_copy_clone(self):
        """Tests shallow_copy and clone of encrypted tensors."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            # test shallow_copy
            encrypted_tensor_shallow = encrypted_tensor.shallow_copy()
            self.assertEqual(
                id(encrypted_tensor_shallow._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_shallow, tensor, "shallow_copy failed")
            # test clone
            encrypted_tensor_clone = encrypted_tensor.clone()
            self.assertNotEqual(
                id(encrypted_tensor_clone._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_clone, tensor, "clone failed")

    def test_copy_(self):
        """Tests copy_ function."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor1 = self._get_random_test_tensor(size=size, is_float=True)
            tensor2 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = MPCTensor(tensor1)
            encrypted_tensor2 = MPCTensor(tensor2)
            encrypted_tensor1.copy_(encrypted_tensor2)
            self._check(encrypted_tensor1, tensor2, "copy_ failed")

    def test_index_select(self):
        """Tests index_select of encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            indices = [[0], [0, 3], [0, 2, 4]]

            for dim in range(tensor.dim()):
                for index in indices:
                    index_tensor = torch.tensor(
                        index, dtype=torch.long, device=self.device
                    )
                    reference = tensor.index_select(dim, index_tensor)
                    encrypted_out = encrypted_tensor.index_select(dim, index_tensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "index_select failed at dim {dim} and index {index}",
                    )

    def test_narrow(self):
        """Tests narrow function."""
        sizes = [(5, 6), (5, 6, 7), (6, 7, 8, 9)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encr_tensor = MPCTensor(tensor)
            for dim in range(len(size)):
                for start in range(size[dim] - 2):
                    for length in range(1, size[dim] - start):
                        tensor_narrow = tensor.narrow(dim, start, length)
                        encr_tensor_narrow = encr_tensor.narrow(dim, start, length)
                        self._check(
                            encr_tensor_narrow,
                            tensor_narrow,
                            "narrow failed along dimension %d" % dim,
                        )

    def test_repeat_expand(self):
        """Tests repeat and expand of encrypted tensors."""
        sizes = [(1, 8), (4, 1, 8)]
        repeat_dims = [(4, 2, 1), (4, 2, 10)]
        expand_dims = [(4, 2, 8), (4, 5, 8), (10, 4, 5, 8)]

        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            for dims in repeat_dims:
                encrypted_tensor_repeated = encrypted_tensor.repeat(*dims)
                # test that repeat copies tensor's data
                self.assertNotEqual(
                    id(encrypted_tensor_repeated._tensor), id(encrypted_tensor._tensor)
                )
                self._check(
                    encrypted_tensor_repeated,
                    tensor.repeat(*dims),
                    f"repeat failed with dims {dims}",
                )

            for dims in expand_dims:
                encrypted_tensor_expanded = encrypted_tensor.expand(*dims)
                # test that expand creates a view into the same underlying tensor
                self.assertNotEqual(
                    id(encrypted_tensor_expanded.share), id(encrypted_tensor.share)
                )
                self._check(
                    encrypted_tensor_expanded,
                    tensor.expand(*dims),
                    f"repeat failed with dims {dims}",
                )

    def test_view_flatten(self):
        """Tests view and flatten of encrypted tensors."""
        sizes = [(100,), (4, 25), (2, 5, 10)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            for dim in range(tensor.dim()):
                self._check(
                    encrypted_tensor.flatten(start_dim=dim),
                    tensor.flatten(start_dim=dim),
                    f"flatten failed with dim {dim}",
                )

            shapes = [100, (5, 20), (10, 2, 5), (-1, 10)]
            for shape in shapes:
                self._check(
                    encrypted_tensor.view(shape),
                    tensor.view(shape),
                    f"view failed with shape {shape}",
                )

    def test_roll(self):
        """Tests roll of encrypted tensors."""
        sizes = [(10, 1), (5, 2), (5, 10, 15)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            roll_shifts = [1, 2, 3, (2, 1)]
            roll_dims = [0, 1, 0, (0, 1)]

            for shifts, dims in zip(roll_shifts, roll_dims):
                encrypted_tensor_rolled = encrypted_tensor.roll(shifts, dims=dims)
                self.assertEqual(encrypted_tensor_rolled.numel(), tensor.numel())
                self._check(
                    encrypted_tensor_rolled,
                    tensor.roll(shifts, dims=dims),
                    f"roll failed with shift {shifts} and dims {dims}",
                )

    def test_unfold(self):
        """Tests unfold of encrypted tensors."""
        tensor_sizes = [(8,), (15, 10, 5), (5, 10, 15, 20)]
        for tensor_size in tensor_sizes:
            tensor = self._get_random_test_tensor(size=tensor_size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            for size, step in itertools.product(range(1, 4), range(1, 4)):
                # check unfold along higher dimension if possible
                for dim in range(tensor.dim()):
                    self._check(
                        encrypted_tensor.unfold(dim, size, step),
                        tensor.unfold(dim, size, step),
                        "unfold failed with dim "
                        f"{dim}, size {size}, and step {step}",
                    )

    def test_to(self):
        """Tests Arithemetic/Binary SharedTensor type conversions."""
        from crypten.mpc.ptype import ptype as Ptype

        tensor_sizes = [(), (1,), (5,), (1, 1), (5, 5), (1, 1, 1), (5, 5, 5)]

        for size in tensor_sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)

            binary_encrypted_tensor = encrypted_tensor.to(Ptype.binary)
            self.assertEqual(binary_encrypted_tensor.ptype, Ptype.binary)

            # check original encrypted_tensor was not modified after conversion
            self._check(
                encrypted_tensor,
                tensor,
                "encrypted_tensor was modified during conversion to BinarySharedTensor.",
            )

            encrypted_from_binary = binary_encrypted_tensor.to(Ptype.arithmetic)
            self._check(
                encrypted_from_binary,
                tensor,
                "to failed from BinarySharedTensor to ArithmeticSharedTensor",
            )

        # Test API
        tensor = self._get_random_test_tensor(size=(5,), is_float=True)
        encrypted_tensor = MPCTensor(tensor)

        if torch.cuda.is_available():
            encrypted_tensor = encrypted_tensor.to("cuda")
            self.assertEqual(encrypted_tensor.device.type, "cuda")
            self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)
            self._check(
                encrypted_tensor,
                tensor,
                "encrypted_tensor was modified during conversion to cuda",
            )

            encrypted_tensor = encrypted_tensor.to(device="cuda")
            self.assertEqual(encrypted_tensor.device.type, "cuda")
            self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)
            self._check(
                encrypted_tensor,
                tensor,
                "encrypted_tensor was modified during conversion to cuda",
            )

        encrypted_tensor = encrypted_tensor.to("cpu")
        self.assertEqual(encrypted_tensor.device.type, "cpu")
        self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)
        self._check(
            encrypted_tensor,
            tensor,
            "encrypted_tensor was modified during conversion to cpu",
        )

        encrypted_tensor = encrypted_tensor.to(device="cpu")
        self.assertEqual(encrypted_tensor.device.type, "cpu")
        self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)
        self._check(
            encrypted_tensor,
            tensor,
            "encrypted_tensor was modified during conversion to cpu",
        )

        encrypted_tensor = encrypted_tensor.to(ptype=Ptype.binary)
        self.assertEqual(encrypted_tensor.device.type, "cpu")
        self.assertEqual(encrypted_tensor.ptype, Ptype.binary)
        self._check(
            encrypted_tensor,
            tensor,
            "encrypted_tensor was modified during conversion to BinarySharedTensor.",
        )

        encrypted_tensor = encrypted_tensor.to(ptype=Ptype.arithmetic)
        self.assertEqual(encrypted_tensor.device.type, "cpu")
        self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)
        self._check(
            encrypted_tensor,
            tensor,
            "encrypted_tensor was modified during conversion to ArithmeticSharedTensor.",
        )

    def test_cumsum(self):
        """Tests cumulative sum on encrypted tensors."""
        sizes = [(8,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            for dim in range(tensor.dim()):
                self._check(
                    encrypted_tensor.cumsum(dim),
                    tensor.cumsum(dim),
                    f"cumsum failed along {dim} dim",
                )

    def test_trace(self):
        """Tests trace operation on 2D encrypted tensors."""
        sizes = [(3, 3), (10, 10), (2, 3)]

        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            self._check(encrypted_tensor.trace(), tensor.trace(), "trace failed")

    def test_flip(self):
        """Tests flip operation on encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            flip_dims = [(0,), (0, 1), (0, 1, 2)]

            for dims in flip_dims:
                if len(dims) <= tensor.dim():
                    self._check(
                        encrypted_tensor.flip(dims),
                        tensor.flip(dims),
                        f"flip failed with {dims} dims",
                    )

    def test_control_flow_failure(self):
        """Tests that control flow fails as expected"""
        tensor = self._get_random_test_tensor(is_float=True)
        encrypted_tensor = MPCTensor(tensor)
        with self.assertRaises(RuntimeError):
            if encrypted_tensor:
                pass

        with self.assertRaises(RuntimeError):
            tensor = 5 if encrypted_tensor else 0

        with self.assertRaises(RuntimeError):
            if False:
                pass
            elif encrypted_tensor:
                pass

    def test_where(self):
        """Tests where() conditional element selection"""
        sizes = [(10,), (5, 10), (1, 5, 10)]
        y_types = [lambda x: x, MPCTensor]

        for size, y_type in itertools.product(sizes, y_types):
            tensor1 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = MPCTensor(tensor1)
            tensor2 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                self._get_random_test_tensor(max_value=1, size=size, is_float=False) + 1
            )
            condition_encrypted = MPCTensor(condition_tensor)
            condition_bool = condition_tensor.bool()

            reference_out = tensor1.where(condition_bool, tensor2)

            encrypted_out = encrypted_tensor1.where(condition_bool, encrypted_tensor2)
            y_is_private = y_type == MPCTensor
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with public condition",
            )

            encrypted_out = encrypted_tensor1.where(
                condition_encrypted, encrypted_tensor2
            )
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with private condition",
            )

            # test scalar y
            scalar = self._get_random_test_tensor(max_value=0, size=[1], is_float=True)
            self._check(
                encrypted_tensor1.where(condition_bool, scalar),
                tensor1.where(condition_bool, scalar),
                "where failed against scalar y with public condition",
            )

            self._check(
                encrypted_tensor1.where(condition_encrypted, scalar),
                tensor1.where(condition_bool, scalar),
                "where failed against scalar y with private condition",
            )

    def test_unbind(self):
        """Tests unbind"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted = MPCTensor(tensor)
            for dim in range(tensor.dim()):
                reference = tensor.unbind(dim)
                encrypted_out = encrypted.unbind(dim)

                self._check_tuple(encrypted_out, reference, "unbind failed")

    def test_split(self):
        """Tests split"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted = MPCTensor(tensor)
            for dim in range(tensor.dim()):
                # Get random split
                split = self._get_random_test_tensor(
                    size=(), max_value=tensor.size(dim)
                )
                split = split.abs().clamp(0, tensor.size(dim) - 1)
                split = split.item()

                # Test int split
                int_split = 1 if split == 0 else split
                reference = tensor.split(int_split, dim=dim)
                encrypted_out = encrypted.split(int_split, dim=dim)
                self._check_tuple(encrypted_out, reference, "split failed")

                # Test list split
                split = [split, tensor.size(dim) - split]
                reference = tensor.split(split, dim=dim)
                encrypted_out = encrypted.split(split, dim=dim)
                self._check_tuple(encrypted_out, reference, "split failed")

    def test_set(self):
        """Tests set correctly re-assigns encrypted shares"""
        sizes = [(1, 5), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor1 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted1 = MPCTensor(tensor1)

            tensor2 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted2 = MPCTensor(tensor2)

            # check encrypted set
            encrypted1.set(encrypted2)
            self._check(
                encrypted1, tensor2, f"set with encrypted other failed with size {size}"
            )

            # check plain text set
            encrypted1 = MPCTensor(tensor1)
            encrypted1.set(tensor2)
            self._check(
                encrypted1,
                tensor2,
                f"set with unencrypted other failed with size {size}",
            )

    def test_polynomial(self):
        """Tests polynomial function"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, max_value=3, is_float=True)
            encrypted = MPCTensor(tensor)
            for terms in range(1, 5):
                coeffs = self._get_random_test_tensor(
                    size=(terms,), max_value=3, is_float=True
                )

                reference = torch.zeros(size=tensor.size(), device=self.device)
                for i, term in enumerate(coeffs.tolist()):
                    reference += term * tensor.pow(i + 1)

                # Test list coeffs
                encrypted_out = encrypted.polynomial(coeffs.tolist())
                self._check(encrypted_out, reference, "polynomial failed")

                # Test plaintext tensor coeffs
                encrypted_out = encrypted.polynomial(coeffs)
                self._check(encrypted_out, reference, "polynomial failed")

                # Test encrypted tensor coeffs
                coeffs_enc = MPCTensor(coeffs)
                encrypted_out = encrypted.polynomial(coeffs_enc)
                self._check(encrypted_out, reference, "polynomial failed")

    def test_gather(self):
        """Test gather function of encrypted tensor"""
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for size in sizes:
            for dim in range(len(size)):
                tensor = self._get_random_test_tensor(size=size, is_float=True)
                index = self._get_random_test_tensor(size=size, is_float=False)
                index = index.abs().clamp(0, 4)
                encrypted = MPCTensor(tensor)
                reference = tensor.gather(dim, index)
                encrypted_out = encrypted.gather(dim, index)
                self._check(encrypted_out, reference, f"gather failed with size {size}")

    def test_dropout(self):
        """
        Tests the dropout functions. Directly compares the zero and non-zero
        entries of the input tensor, since we cannot force the encrypted and
        unencrypted versions to generate identical random output. Also confirms
        that the number of zeros in the encrypted dropout function is as expected.
        """
        all_prob_values = [x * 0.2 for x in range(5)]

        def get_first_nonzero_value(x):
            x = x.flatten()
            x = x[x.abs().ge(1e-4)]
            x = x.take(torch.tensor(0))
            return x

        # check that the encrypted and plaintext versions scale
        # identically, by testing on all-ones tensor
        for prob in all_prob_values:
            tensor = torch.ones([10, 10, 10], device=self.device).float()
            encr_tensor = MPCTensor(tensor)
            dropout_encr = encr_tensor.dropout(prob, training=True)
            dropout_decr = dropout_encr.get_plain_text()
            dropout_plain = F.dropout(tensor, prob, training=True)

            # All non-zero values should be identical in both tensors, so
            # compare any one of them
            decr_nonzero_value = get_first_nonzero_value(dropout_decr)
            plaintext_nonzero_value = get_first_nonzero_value(dropout_plain)

            self.assertTrue(
                math.isclose(
                    decr_nonzero_value,
                    plaintext_nonzero_value,
                    rel_tol=1e-2,
                    abs_tol=1e-2,
                )
            )

        for dropout_fn in ["dropout", "_feature_dropout"]:
            for prob in all_prob_values:
                for size in [(5, 10), (5, 10, 15), (5, 10, 15, 20)]:
                    for inplace in [False, True]:
                        for training in [False, True]:
                            tensor = self._get_random_test_tensor(
                                size=size, ex_zero=True, min_value=1.0, is_float=True
                            )
                            encr_tensor = MPCTensor(tensor)
                            dropout_encr = getattr(encr_tensor, dropout_fn)(
                                prob, inplace=inplace, training=training
                            )
                            if training:
                                # Check the scaling for non-zero elements
                                dropout_decr = dropout_encr.get_plain_text()
                                scaled_tensor = tensor / (1 - prob)
                                reference = dropout_decr.where(
                                    dropout_decr == 0, scaled_tensor
                                )
                            else:
                                reference = tensor
                            self._check(
                                dropout_encr,
                                reference,
                                f"dropout failed with size {size} and probability "
                                f"{prob}",
                            )
                            if inplace:
                                self._check(
                                    encr_tensor,
                                    reference,
                                    f"in-place dropout failed with size {size} and "
                                    f"probability {prob}",
                                )
                            else:
                                self._check(
                                    encr_tensor,
                                    tensor,
                                    "out-of-place dropout modifies input",
                                )
                            # Check that channels that are zeroed are all zeros
                            if dropout_fn in [
                                "dropout2d",
                                "dropout3d",
                                "feature_dropout",
                            ]:
                                dropout_encr_flat = dropout_encr.flatten(
                                    start_dim=0, end_dim=1
                                )
                                dropout_flat = dropout_encr_flat.get_plain_text()
                                for i in range(0, dropout_flat.size(0)):
                                    all_zeros = (dropout_flat[i] == 0).all()
                                    all_nonzeros = (dropout_flat[i] != 0).all()
                                    self.assertTrue(
                                        all_zeros or all_nonzeros,
                                        f"{dropout_fn} failed for size {size} with "
                                        f"training {training} and inplace {inplace}",
                                    )

        # Check the expected number of zero elements
        # For speed, restrict test to single p = 0.4
        encr_tensor = MPCTensor(torch.empty((int(1e5), 2, 2)).fill_(1).to(self.device))
        dropout_encr = encr_tensor.dropout(0.4)
        dropout_tensor = dropout_encr.get_plain_text()
        frac_zero = float((dropout_tensor == 0).sum()) / dropout_tensor.nelement()
        self.assertTrue(math.isclose(frac_zero, 0.4, rel_tol=1e-2, abs_tol=1e-2))

    def _test_cache_save_load(self):
        # Determine expected filepaths
        provider = crypten.mpc.get_default_provider()
        request_path = provider._DEFAULT_CACHE_PATH + f"/request_cache-{self.rank}"
        tuple_path = provider._DEFAULT_CACHE_PATH + f"/tuple_cache-{self.rank}"

        # Clear any existing files in the cache location
        if os.path.exists(request_path):
            os.remove(request_path)
        if os.path.exists(tuple_path):
            os.remove(tuple_path)

        # Store cache values for reference
        requests = provider.request_cache
        tuple_cache = provider.tuple_cache

        # Save cache to file
        provider.save_cache()

        # Assert cache files exist
        self.assertTrue(
            os.path.exists(request_path), "request_cache file not found after save"
        )
        self.assertTrue(
            os.path.exists(tuple_path), "tuple_cache file not found after save"
        )

        # Assert cache empty
        self.assertEqual(
            len(provider.request_cache), 0, "cache save did not clear request cache"
        )
        self.assertEqual(
            len(provider.tuple_cache), 0, "cache save did not clear tuple cache"
        )

        # Ensure test is working properly by not clearing references
        self.assertTrue(len(requests) > 0, "reference requests cleared during save")
        self.assertTrue(len(tuple_cache) > 0, "reference tuples cleared during save")

        # Load cache from file
        provider.load_cache()

        # Assert files are deleted
        self.assertFalse(
            os.path.exists(request_path), "request_cache filepath exists after load"
        )
        self.assertFalse(
            os.path.exists(tuple_path), "tuple_cache filepath exists after load"
        )

        # Assert request cache is loaded as expected
        self.assertEqual(
            provider.request_cache, requests, "loaded request_cache is incorrect"
        )

        # Assert loaded tuple dict is as expected
        tc = [(k, v) for k, v in provider.tuple_cache.items()]
        ref = [(k, v) for k, v in tuple_cache.items()]
        for i in range(len(tc)):
            t, r = tc[i], ref[i]
            t_key, r_key = t[0], r[0]
            t_tuples, r_tuples = t[1], r[1]

            # Check keys
            self.assertEqual(t_key, r_key, "Loaded tuple_cache key is incorrect")

            # Check tuple values
            for j in range(len(t_tuples)):
                t_tuple, r_tuple = t_tuples[j], r_tuples[j]
                for k in range(len(t_tuple)):
                    t_tensor = t_tuple[k]._tensor
                    r_tensor = r_tuple[k]._tensor
                    self.assertTrue(
                        t_tensor.eq(r_tensor).all(),
                        "Loaded tuple_cache tuple tensor incorrect",
                    )

    def test_tuple_cache(self):
        # Skip RSS setting since it does not generate tuples
        if cfg.mpc.protocol == "replicated":
            return

        # TODO: encorporate wrap_rng for 3PC+ settings
        if comm.get().get_world_size() > 2:
            return

        provider = crypten.mpc.get_default_provider()

        # Test tracing attribute
        crypten.trace()
        self.assertTrue(provider.tracing)

        x = get_random_test_tensor(is_float=True)
        x = crypten.cryptensor(x)

        _ = x.square()
        _ = x * x
        _ = x.matmul(x.t())
        _ = x.relu()
        y = x.unsqueeze(0)
        _ = y.conv1d(y, stride=2)

        # Populate reference requests
        ref_names = ["square"]
        ref_names += ["generate_additive_triple"] * 2
        ref_names += ["generate_binary_triple"] * 7 + ["B2A_rng"]
        ref_names += ["generate_additive_triple"] * 2

        ref_args = [
            (torch.Size([1, 5]),),
            (torch.Size([1, 5]), torch.Size([1, 5]), "mul"),
            (torch.Size([1, 5]), torch.Size([5, 1]), "matmul"),
            (torch.Size([1, 1, 5]), torch.Size([1, 1, 5])),
        ]
        ref_args += [(torch.Size([2, 1, 1, 5]), torch.Size([2, 1, 1, 5]))] * 6
        ref_args += [(torch.Size([1, 5]),)]
        ref_args += [(torch.Size([1, 5]), torch.Size([1, 5]), "mul")]
        ref_args += [(torch.Size([1, 1, 5]), torch.Size([1, 1, 5]), "conv1d")]

        kwargs = {"device": torch.device("cpu")}
        conv_kwargs = {"device": torch.device("cpu"), "stride": 2}
        requests = [(ref_names[i], ref_args[i], kwargs) for i in range(12)]
        requests += [(ref_names[12], ref_args[12], conv_kwargs)]

        self.assertEqual(
            provider.request_cache,
            requests,
            "TupleProvider request cache incorrect",
        )

        crypten.trace(False)
        self.assertFalse(provider.tracing)

        # Check that cache populates as expected
        crypten.fill_cache()
        kwargs = frozenset(kwargs.items())
        conv_kwargs = frozenset(conv_kwargs.items())

        keys = [(ref_names[i], ref_args[i], kwargs) for i in range(12)]
        keys += [(ref_names[12], ref_args[12], conv_kwargs)]

        self.assertEqual(
            set(provider.tuple_cache.keys()),
            set(keys),
            "TupleProvider tuple_cache populated incorrectly",
        )

        # Test saving from / loading to cache
        self._test_cache_save_load()

        # Test that function calls return from cache when trace is off
        crypten.trace(False)
        _ = x.square()
        _ = x * x
        _ = x.matmul(x.t())
        _ = x.relu()
        y = x.unsqueeze(0)
        _ = y.conv1d(y, stride=2)

        for v in provider.tuple_cache.values():
            self.assertEqual(
                len(v), 0, msg="TupleProvider is not popping tuples properly from cache"
            )


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestMPC):
    def setUp(self):
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self):
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestMPC):
    def setUp(self):
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self):
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTTP, self).tearDown()


class Test3PC(MultiProcessTestCase, TestMPC):
    def setUp(self):
        super(Test3PC, self).setUp(world_size=3)

    def tearDown(self):
        super(Test3PC, self).tearDown()


class TestRSS(MultiProcessTestCase, TestMPC):
    def setUp(self):
        self._original_protocol = cfg.mpc.protocol
        cfg.mpc.protocol = "replicated"
        super(TestRSS, self).setUp(world_size=3)

    def tearDown(self):
        cfg.mpc.protocol = self._original_protocol
        super(TestRSS, self).tearDown()


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target of another test)
if __name__ == "__main__":
    unittest.main()
