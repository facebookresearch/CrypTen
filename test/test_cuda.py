#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import math
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor, is_int_tensor
from crypten.cuda import CUDALongTensor
from crypten.mpc import ConfigManager, MPCTensor, ptype as Ptype
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor, beaver

from .test_mpc import TestMPC


class TestCUDA(TestMPC):
    """
        This class tests all functions of CUDALongTensor as well as its integration with MPCTensor.
    """

    def _check_int(self, result, reference, msg):
        # Check sizes match
        self.assertTrue(result.size() == reference.size(), msg)
        self.assertTrue(is_int_tensor(result), "result must be a int tensor")
        self.assertTrue(is_int_tensor(reference), "reference must be a int tensor")

        is_eq = (result == reference).all().item() == 1

        if not is_eq:
            logging.info(msg)
            logging.info("Result %s" % result)
            logging.info("Reference %s" % reference)
            logging.info("Result - Reference = %s" % (result - reference))

        self.assertTrue(is_eq, msg=msg)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_patched_matmul(self):
        x = get_random_test_tensor(max_value=2 ** 62, is_float=False)
        x_cuda = CUDALongTensor(x.cuda())
        for width in range(2, x.nelement()):
            matrix_size = (x.nelement(), width)
            y = get_random_test_tensor(
                size=matrix_size, max_value=2 ** 62, is_float=False
            )

            y_cuda = CUDALongTensor(y.cuda())
            z = torch.matmul(x_cuda, y_cuda).data

            reference = torch.matmul(x, y)
            self._check_int(z.cpu(), reference, "matmul failed for cuda_patches")

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_smaller_signal_one_channel(self):
        self._patched_conv1d(5, 1)
        self._conv1d(5, 1)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_smaller_signal_many_channels(self):
        self._patched_conv1d(5, 5)
        self._conv1d(5, 5)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_larger_signal_one_channel(self):
        self._patched_conv1d(16, 1)
        self._conv1d(16, 1)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_larger_signal_many_channels(self):
        self._patched_conv1d(16, 5)
        self._conv1d(16, 5)

    def _patched_conv1d(self, signal_size, in_channels):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
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
                signal = get_random_test_tensor(size=input_size, is_float=False)

                if func_name == "conv1d":
                    k_size = (out_channels, in_channels, kernel_size)
                else:
                    k_size = (in_channels, out_channels, kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                signal_cuda = CUDALongTensor(signal.cuda())
                kernel_cuda = CUDALongTensor(kernel.cuda())

                reference = getattr(F, func_name)(
                    signal, kernel, padding=padding, stride=stride
                )
                result = getattr(F, func_name)(
                    signal_cuda, kernel_cuda, padding=padding, stride=stride
                )
                result = result.data.cpu()

                self._check_int(result, reference, f"{func_name} failed")

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv2d_square_image_one_channel(self):
        self._patched_conv2d((5, 5), 1)
        self._conv2d((5, 5), 1)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv2d_square_image_many_channels(self):
        self._patched_conv2d((5, 5), 5)
        self._conv2d((5, 5), 5)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv2d_rectangular_image_one_channel(self):
        self._patched_conv2d((16, 7), 1)
        self._conv2d((16, 7), 1)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv2d_rectangular_image_many_channels(self):
        self._patched_conv2d((16, 7), 5)
        self._conv2d((16, 7), 5)

    def _patched_conv2d(self, image_size, in_channels):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
        nbatches = [1, 3]
        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        ochannels = [1, 3, 6]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]

        for func_name in ["conv2d"]:
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
                input = get_random_test_tensor(size=input_size, is_float=False)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels, in_channels, *kernel_size)
                else:
                    k_size = (in_channels, out_channels, *kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                input_cuda = CUDALongTensor(input.cuda())
                kernel_cuda = CUDALongTensor(kernel.cuda())

                result = getattr(F, func_name)(
                    input_cuda, kernel_cuda, padding=padding, stride=stride
                )
                result = result.data.cpu()

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input, kernel, padding=padding, stride=stride
                )
                self._check_int(result, reference, "%s failed" % func_name)


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestCUDA):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.device = "cuda"

    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.CrypTensor.set_grad_enabled(False)
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedFirstParty)
        super(TestTFP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestCUDA):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.device = "cuda"

    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.CrypTensor.set_grad_enabled(False)
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)
        super(TestTTP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target of another test)
if __name__ == "__main__":
    unittest.main()
