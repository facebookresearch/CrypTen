#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from test.multiprocess_test_case import MultiProcessTestCase

import crypten
from crypten.debug import configure_logging, pdb, set_debug_mode
from torch import tensor


class TestDebug(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communicator
        if self.rank >= 0:
            crypten.init()

        # Testing debug mode
        set_debug_mode()

    def testLogging(self):
        configure_logging()

    def testPdb(self):
        self.assertTrue(hasattr(pdb, "set_trace"))

    def test_wrap_error_detection(self):
        """Force a wrap error and test whether it raises in debug mode."""
        encrypted_tensor = crypten.cryptensor(0)
        encrypted_tensor.share = tensor(2 ** 63 - 1)
        with self.assertRaises(ValueError):
            encrypted_tensor.div(2)
