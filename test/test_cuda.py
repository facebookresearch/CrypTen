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
