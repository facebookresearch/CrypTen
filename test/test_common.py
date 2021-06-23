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

    def test_encode_decode(self):
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
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedFirstParty)
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

    def test_nearest_integer_division(self):
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

    def test_chebyshev_series(self):
        """Checks coefficients returned by chebyshev_series are correct"""
        for width, terms in [(6, 10), (6, 20)]:
            result = chebyshev_series(torch.tanh, width, terms)
            # check shape
            self.assertTrue(result.shape == torch.Size([terms]))
            # check terms
            self.assertTrue(result[0] < 1e-4)
            self.assertTrue(torch.isclose(result[-1], torch.tensor(3.5e-2), atol=1e-1))

    def test_config_managers(self):
        """Checks setting configuartion with config manager works"""
        # Set the config directly
        cfgs = [
            (crypten.common.functions.approximations, "exp_iterations", "ApproxConfig"),
            (crypten.mpc, "max_method", "MPCConfig"),
        ]

        for cfg in cfgs:
            base = cfg[0]
            arg_name = cfg[1]
            cfg_name = cfg[2]

            setattr(base.config, arg_name, 8)
            self.assertTrue(getattr(base.config, arg_name) == 8)

            # Set with a context manager
            with base.ConfigManager(arg_name, 3):
                self.assertTrue(getattr(base.config, arg_name) == 3)
            self.assertTrue(getattr(base.config, arg_name) == 8)

            kwargs = {arg_name: 5}
            new_config = getattr(base, cfg_name)(**kwargs)

            base.set_config(new_config)
            self.assertTrue(getattr(base.config, arg_name) == 5)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
