#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import torch.nn.functional as F
from crypten.common.tensor_types import is_float_tensor
from crypten.mpc.primitives import ArithmeticSharedTensor, resharing


class TestRSSArithmetic(MultiProcessTestCase):
    """
        This class tests all functions of the ArithmeticSharedTensor.
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp(world_size=3)
        # We don't want the main process (rank -1) to initialize the communcator
        if self.rank >= 0:
            crypten.init()

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

    def test_square(self):
        tensor = get_random_test_tensor(is_float=True)
        reference = tensor * tensor
        encrypted = ArithmeticSharedTensor(tensor)
        encrypted_out = resharing.square(encrypted)
        encrypted_out = encrypted_out.div_(encrypted.encoder.scale)
        self._check(encrypted_out, reference, "square failed")

    def test_matmul(self):
        """Test matrix multiplication."""
        tensor = get_random_test_tensor(max_value=7, is_float=True)
        for width in range(2, tensor.nelement()):
            matrix_size = (tensor.nelement(), width)
            matrix = get_random_test_tensor(
                max_value=7, size=matrix_size, is_float=True
            )
            reference = tensor.matmul(matrix)
            encrypted_tensor = ArithmeticSharedTensor(tensor)
            matrix = ArithmeticSharedTensor(matrix)
            encrypted_tensor = resharing.matmul(encrypted_tensor, matrix)
            encrypted_tensor = encrypted_tensor.div_(matrix.encoder.scale)

            self._check(
                encrypted_tensor,
                reference,
                "Private-private matrix multiplication failed",
            )

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

        for func_name in ["conv1d", "conv_transpose1d"]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
            ) in itertools.product(
                nbatches, kernel_sizes, ochannels, paddings, strides
            ):
                input_size = (batches, in_channels, signal_size)
                signal = get_random_test_tensor(size=input_size, is_float=True)

                if func_name == "conv1d":
                    k_size = (out_channels, in_channels, kernel_size)
                else:
                    k_size = (in_channels, out_channels, kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=True)

                reference = getattr(F, func_name)(
                    signal, kernel, padding=padding, stride=stride
                )
                encrypted_signal = ArithmeticSharedTensor(signal)
                encrypted_kernel = ArithmeticSharedTensor(kernel)
                encrypted_conv = getattr(resharing, func_name)(
                    encrypted_signal, encrypted_kernel, padding=padding, stride=stride
                )

                encrypted_conv = encrypted_conv.div_(encrypted_signal.encoder.scale)

                self._check(encrypted_conv, reference, f"{func_name} failed")

    def test_conv2d_square_image_one_channel(self):
        self._conv2d((5, 5), 1)

    def test_conv2d_square_image_many_channels(self):
        self._conv2d((5, 5), 5)

    def test_conv2d_rectangular_image_one_channel(self):
        self._conv2d((16, 7), 1)

    def test_conv2d_rectangular_image_many_channels(self):
        self._conv2d((16, 7), 5)

    def _conv2d(self, image_size, in_channels):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        ochannels = [1, 3, 6]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]

        for func_name in ["conv2d", "conv_transpose2d"]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
            ) in itertools.product(
                nbatches, kernel_sizes, ochannels, paddings, strides
            ):

                # sample input:
                input_size = (batches, in_channels, *image_size)
                input = get_random_test_tensor(size=input_size, is_float=True)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels, in_channels, *kernel_size)
                else:
                    k_size = (in_channels, out_channels, *kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=True)

                # perform filtering:
                encr_matrix = ArithmeticSharedTensor(input)
                encr_kernel = ArithmeticSharedTensor(kernel)
                encr_conv = getattr(resharing, func_name)(
                    encr_matrix, encr_kernel, padding=padding, stride=stride
                )

                encr_conv = encr_conv.div_(encr_matrix.encoder.scale)

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input, kernel, padding=padding, stride=stride
                )
                self._check(encr_conv, reference, "%s failed" % func_name)
