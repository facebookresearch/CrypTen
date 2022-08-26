#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch
from crypten.config import cfg
from crypten.debug import configure_logging, pdb
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestDebug(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communicator
        if self.rank >= 0:
            crypten.init()

            # Testing debug mode
            cfg.debug.debug_mode = True
            cfg.debug.validation_mode = True

    def testLogging(self):
        configure_logging()

    def testPdb(self):
        self.assertTrue(hasattr(pdb, "set_trace"))

    def test_wrap_error_detection(self):
        """Force a wrap error and test whether it raises in debug mode."""
        encrypted_tensor = crypten.cryptensor(0)
        encrypted_tensor.share = torch.tensor(2**63 - 1)
        with self.assertRaises(ValueError):
            encrypted_tensor.div(2)

    def test_correctness_validation(self):
        for grad_enabled in [False, True]:
            crypten.set_grad_enabled(grad_enabled)

            tensor = get_random_test_tensor(size=(2, 2), is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            # Test properties (non-tensor outputs)
            _ = encrypted_tensor.size()

            # Ensure correct validation works properly
            encrypted_tensor.add(1)

            # Ensure incorrect validation works properly for size
            encrypted_tensor.add = lambda y: crypten.cryptensor(0)
            with self.assertRaises(ValueError):
                encrypted_tensor.add(10)

            # Ensure incorrect validation works properly for value
            encrypted_tensor.add = lambda y: crypten.cryptensor(tensor)
            with self.assertRaises(ValueError):
                encrypted_tensor.add(10)

            # Test matmul in validation mode
            x = get_random_test_tensor(size=(3, 5), is_float=True)
            y = get_random_test_tensor(size=(1, 3), is_float=True)

            x = crypten.cryptensor(x)
            y = crypten.cryptensor(y)
            _ = y.matmul(x)
