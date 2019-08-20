#!/usr/bin/env python3

import sys
import unittest

import crypten
import torch
from crypten.common.encoder import FixedPointEncoder, nearest_integer_division


def get_test_tensor(max_value=10, float=False):
    """Create simple test tensor."""
    tensor = torch.LongTensor([value for value in range(max_value)])
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
        fpe = FixedPointEncoder(precision_bits=16)
        for float in [False, True]:
            tensor = get_test_tensor(float=float)
            decoded = fpe.decode(fpe.encode(tensor), to_float=float)
            self._check(
                decoded,
                tensor,
                "Encoding/decoding a %s failed." % "float" if float else "long",
            )

        # Make sure encoding a subclass of EncryptedTensor is a no-op
        tensor = get_test_tensor(float=True)
        encrypted_tensor = crypten.MPCTensor(tensor)
        encrypted_tensor = fpe.encode(encrypted_tensor)
        self._check(
            encrypted_tensor.get_plain_text(),
            tensor,
            "Encoding an EncryptedTensor failed.",
        )

        # Try a few other types.
        for dtype in [torch.uint8, torch.int8, torch.int16]:
            tensor = torch.zeros(5, dtype=dtype).random_()
            decoded = fpe.decode(fpe.encode(tensor), to_float=False).type(dtype)
            self._check(decoded, tensor, "Encoding/decoding a %s failed." % dtype)

    def test_nearest_integer_division(self):
        # test without scaling:
        scale = 1
        reference = [[-26, -25, -7, -5, -4, -1, 0, 1, 3, 4, 5, 7, 25, 26]]
        tensor = torch.LongTensor(reference)
        result = nearest_integer_division(tensor, scale)
        self._check(
            torch.LongTensor(result.tolist()),
            torch.LongTensor(reference),
            "Nearest integer division failed.",
        )

        # test with scaling:
        scale = 4
        reference = [[-6, -6, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 6, 6]]
        result = nearest_integer_division(tensor, scale)
        self._check(
            torch.LongTensor(result.tolist()),
            torch.LongTensor(reference),
            "Nearest integer division failed.",
        )


if __name__ == "__main__":
    unittest.main(argv=sys.argv[0])
