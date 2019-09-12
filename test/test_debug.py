#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crypten.debug import pdb, configure_logging

import unittest


class TestDebug(unittest.TestCase):
    def testLogging(self):
        configure_logging()

    def testPdb(self):
        self.assertTrue(hasattr(pdb, 'set_trace'))
