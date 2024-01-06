#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest

import crypten
import torch
from crypten.common.util import chebyshev_series
from crypten.config import cfg
from crypten.encoder import FixedPointEncoder, nearest_integer_division


def get_test_tensor(max_value=10, float=False):
    """Create simple test tensor."""
    tensor = torch.tensor(list(range(max_value)), dtype=torch.long)
    if float:
        tensor = tensor.float()
    return tensor


class TestCommon(unittest.TestCase):
    """
    Test cases for common functionality.
    """

    def _check(self, tensor, reference, msg):
        test_passed = (tensor == reference).all().item() == 1
        self.assertTrue(test_passed, msg=msg)

    def test_encode_decode(self) -> None:
        """Tests tensor encoding and decoding."""
        for float in [False, True]:
            if float:
                fpe = FixedPointEncoder(precision_bits=16)
            else:
                fpe = FixedPointEncoder(precision_bits=0)
            tensor = get_test_tensor(float=float)
            decoded = fpe.decode(fpe.encode(tensor))
            self._check(
                decoded,
                tensor,
                "Encoding/decoding a %s failed." % "float" if float else "long",
            )

        # Make sure encoding a subclass of CrypTensor is a no-op
        cfg.mpc.provider = "TFP"
        crypten.init()

        tensor = get_test_tensor(float=True)
        encrypted_tensor = crypten.cryptensor(tensor)
        encrypted_tensor = fpe.encode(encrypted_tensor)
        self._check(
            encrypted_tensor.get_plain_text(),
            tensor,
            "Encoding an EncryptedTensor failed.",
        )

        # Try a few other types.
        fpe = FixedPointEncoder(precision_bits=0)
        for dtype in [torch.uint8, torch.int8, torch.int16]:
            tensor = torch.zeros(5, dtype=dtype).random_()
            decoded = fpe.decode(fpe.encode(tensor)).type(dtype)
            self._check(decoded, tensor, "Encoding/decoding a %s failed." % dtype)

    def test_nearest_integer_division(self) -> None:
        # test without scaling:
        scale = 1
        reference = [[-26, -25, -7, -5, -4, -1, 0, 1, 3, 4, 5, 7, 25, 26]]
        tensor = torch.tensor(reference, dtype=torch.long)
        result = nearest_integer_division(tensor, scale)
        self._check(
            torch.tensor(result.tolist(), dtype=torch.long),
            torch.tensor(reference, dtype=torch.long),
            "Nearest integer division failed.",
        )

        # test with scaling:
        scale = 4
        reference = [[-6, -6, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 6, 6]]
        result = nearest_integer_division(tensor, scale)
        self._check(
            torch.tensor(result.tolist(), dtype=torch.long),
            torch.tensor(reference, dtype=torch.long),
            "Nearest integer division failed.",
        )

    def test_chebyshev_series(self) -> None:
        """Checks coefficients returned by chebyshev_series are correct"""
        for width, terms in [(6, 10), (6, 20)]:
            result = chebyshev_series(torch.tanh, width, terms)
            # check shape
            self.assertTrue(result.shape == torch.Size([terms]))
            # check terms
            self.assertTrue(result[0] < 1e-4)
            self.assertTrue(torch.isclose(result[-1], torch.tensor(3.5e-2), atol=1e-1))

    def test_config(self) -> None:
        """Checks setting configuartion with config manager works"""
        # Set the config directly
        crypten.init()

        cfgs = [
            "functions.exp_iterations",
            "functions.max_method",
        ]

        for _cfg in cfgs:
            cfg[_cfg] = 10
            self.assertTrue(cfg[_cfg] == 10, "cfg.set failed")

            # Set with a context manager
            with cfg.temp_override({_cfg: 3}):
                self.assertTrue(cfg[_cfg] == 3, "temp_override failed to set values")
            self.assertTrue(cfg[_cfg] == 10, "temp_override values persist")


if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
