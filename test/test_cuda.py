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
from crypten.cuda import cuda_patches
from crypten.mpc import MPCTensor, ptype as Ptype
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor, beaver


class TestCUDA(object):
    """
        This class tests all functions of MPCTensor.
    """

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
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
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

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_patched_mul(self):
        x = get_random_test_tensor(max_value=2 ** 62, is_float=False)

        # check mul with tensor of same size
        y = get_random_test_tensor(max_value=2 ** 62, is_float=False)
        z = cuda_patches.mul(x.cuda(), y.cuda())
        z = z.cpu()
        reference = torch.mul(x, y)

        self._check_int(z, reference, "matmul failed for cuda_patches")

        # check mul with a scalar
        y = get_random_test_tensor(size=(1,), max_value=2 ** 62, is_float=False)
        z = cuda_patches.mul(x.cuda(), y.cuda())
        z = z.cpu()
        reference = torch.mul(x, y)

        self._check_int(z, reference, "matmul failed for cuda_patches")

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_patched_matmul(self):
        x = get_random_test_tensor(max_value=2 ** 62, is_float=False)

        for width in range(2, x.nelement()):
            matrix_size = (x.nelement(), width)
            y = get_random_test_tensor(
                size=matrix_size, max_value=2 ** 62, is_float=False
            )

            z = cuda_patches.matmul(x.cuda(), y.cuda())
            z = z.cpu()

            reference = torch.matmul(x, y)
            self._check_int(z, reference, "matmul failed for cuda_patches")

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_patched_matmul(self):
        x = get_random_test_tensor(max_value=2 ** 62, is_float=False)

        for width in range(2, x.nelement()):
            matrix_size = (x.nelement(), width)
            y = get_random_test_tensor(
                size=matrix_size, max_value=2 ** 62, is_float=False
            )

            z = cuda_patches.matmul(x.cuda(), y.cuda())
            z = z.cpu()

            reference = torch.matmul(x, y)
            self._check_int(z, reference, "matmul failed for cuda_patches")

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv1d_smaller_signal_one_channel(self):
        self._patched_conv1d(5, 1)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv1d_smaller_signal_many_channels(self):
        self._patched_conv1d(5, 5)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv1d_larger_signal_one_channel(self):
        self._patched_conv1d(16, 1)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv1d_larger_signal_many_channels(self):
        self._patched_conv1d(16, 5)

    def _patched_conv1d(self, signal_size, in_channels):
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
                signal = get_random_test_tensor(size=input_size, is_float=False)

                if func_name == "conv1d":
                    k_size = (out_channels, in_channels, kernel_size)
                else:
                    k_size = (in_channels, out_channels, kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                reference = getattr(F, func_name)(
                    signal, kernel, padding=padding, stride=stride
                )
                result = getattr(cuda_patches, func_name)(
                    signal.cuda(), kernel.cuda(), padding=padding, stride=stride
                )
                result = result.cpu()

                self._check_int(result, reference, f"{func_name} failed")

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv2d_square_image_one_channel(self):
        self._patched_conv2d((5, 5), 1)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv2d_square_image_many_channels(self):
        self._patched_conv2d((5, 5), 5)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv2d_rectangular_image_one_channel(self):
        self._patched_conv2d((16, 7), 1)

    @unittest.skipIf(torch.cuda.is_available() == False, "requires CUDA")
    def test_conv2d_rectangular_image_many_channels(self):
        self._patched_conv2d((16, 7), 5)

    def _patched_conv2d(self, image_size, in_channels):
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
                input = get_random_test_tensor(size=input_size, is_float=False)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels, in_channels, *kernel_size)
                else:
                    k_size = (in_channels, out_channels, *kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                result = getattr(cuda_patches, func_name)(
                    input.cuda(), kernel.cuda(), padding=padding, stride=stride
                )
                result = result.cpu()

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input, kernel, padding=padding, stride=stride
                )
                self._check_int(result, reference, "%s failed" % func_name)


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestCUDA):
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
