#!/usr/bin/env python3

# dependencies:
import logging
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import torch


BinarySharedTensor, is_int_tensor = None, None


def import_crypten():
    """
    Imports CrypTen types. This function is called after environment variables
    in MultiProcessTestCase.setUp() are set, and sets the class references for
    all test functions.
    """
    global BinarySharedTensor
    global is_int_tensor
    from crypten.primitives.binary.binary import (
        BinarySharedTensor as _BinarySharedTensor,
    )
    from crypten.common.tensor_types import is_int_tensor as _is_int_tensor

    BinarySharedTensor = _BinarySharedTensor
    is_int_tensor = _is_int_tensor


class TestBinary(MultiProcessTestCase):
    """
        This class tests all functions of BinarySharedTensor.
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        import_crypten()

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        if self.rank != 0:  # Do not check for non-0 rank
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_int_tensor(reference), "reference must be a long")
        test_passed = (tensor == reference).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

    def test_encrypt_decrypt(self):
        """
            Tests tensor encryption and decryption for both positive
            and negative values.
        """
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            reference = get_random_test_tensor(size=size, is_float=False)
            with self.benchmark(tensor_type="BinarySharedTensor") as bench:
                for _ in bench.iters:
                    encrypted_tensor = BinarySharedTensor(reference)
                    self._check(encrypted_tensor, reference, "en/decryption failed")

    def test_transpose(self):
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor = BinarySharedTensor(tensor)

            if len(size) == 2:  # t() asserts dim == 2
                reference = tensor.t()
                with self.benchmark(niters=10) as bench:
                    for _ in bench.iters:
                        encrypted_out = encrypted_tensor.t()
                self._check(encrypted_out, reference, "t() failed")

            for dim0 in range(len(size)):
                for dim1 in range(len(size)):
                    reference = tensor.transpose(dim0, dim1)
                    with self.benchmark(niters=10) as bench:
                        for _ in bench.iters:
                            encrypted_out = encrypted_tensor.transpose(dim0, dim1)
                    self._check(encrypted_out, reference, "transpose failed")

    def test_XOR(self):
        """Test bitwise-XOR function on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor ^ tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            with self.benchmark(tensor_type=tensor_type.__name__) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor ^ encrypted_tensor2
            self._check(encrypted_out, reference, "%s XOR failed" % tensor_type)

    def test_AND(self):
        """Test bitwise-AND function on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor & tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            with self.benchmark(tensor_type=tensor_type.__name__) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor & encrypted_tensor2
            self._check(encrypted_out, reference, "%s AND failed" % tensor_type)

    def test_comparators(self):
        """Test circuit functions on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            for func in ["add", "gt", "ge", "lt", "le", "eq", "ne"]:
                tensor = get_random_test_tensor(is_float=False)
                tensor2 = get_random_test_tensor(is_float=False)
                reference = getattr(tensor, func)(tensor2).long()

                encrypted_tensor = BinarySharedTensor(tensor)
                encrypted_tensor2 = tensor_type(tensor2)

                with self.benchmark(
                    niters=10, tensor_type=tensor_type.__name__, func=func
                ) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, func)(
                            encrypted_tensor2
                        )

                self._check(
                    encrypted_out,
                    reference,
                    "%s binary %s failed" % (type(tensor2), func),
                )

    # TODO: Fix implementations for BinarySharedTensor
    @unittest.skip
    def test_max_min(self):
        """Test max and min"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (5, 1),
            (1, 1, 1),
            (1, 5, 1),
            (1, 1, 5),
            (1, 5, 5),
            (5, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 1, 1, 1),
            (5, 5, 1, 1),
            (1, 5, 5, 5),
            (5, 5, 5, 5),
        ]
        test_cases = [torch.LongTensor([[1, 1, 2, 1, 4, 1, 3, 4]])] + [
            get_random_test_tensor(size=size, is_float=False) for size in sizes
        ]

        # TODO: Fix implementations for BinarySharedTensor
        for tensor in test_cases:
            encrypted_tensor = BinarySharedTensor(tensor)
            for comp in ["max", "min"]:
                reference = getattr(tensor, comp)()
                with self.benchmark(niters=10, comp=comp, dim=None) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)()
                self._check(encrypted_out, reference, "%s reduction failed" % comp)

                for dim in range(tensor.dim()):
                    reference = getattr(tensor, comp)(dim=dim)[0]
                    with self.benchmark(niters=10, comp=comp, dim=dim) as bench:
                        for _ in bench.iters:
                            encrypted_out = getattr(encrypted_tensor, comp)(dim=dim)

                    self._check(encrypted_out, reference, "%s reduction failed" % comp)

    # TODO: Fix implementations for BinarySharedTensor
    @unittest.skip
    def test_argmax_argmin(self):
        """Test argmax and argmin"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (5, 1),
            (1, 1, 1),
            (1, 5, 1),
            (1, 1, 5),
            (1, 5, 5),
            (5, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 1, 1, 1),
            (5, 5, 1, 1),
            (1, 5, 5, 5),
            (5, 5, 5, 5),
        ]
        test_cases = [torch.LongTensor([[1, 1, 2, 1, 4, 1, 3, 4]])] + [
            get_random_test_tensor(size=size, is_float=False) for size in sizes
        ]

        # TODO: Fix implementations for BinarySharedTensor
        for tensor in test_cases:
            encrypted_tensor = BinarySharedTensor(tensor)
            for comp in ["argmax", "argmin"]:
                cmp = comp[3:]

                # Compute one-hot argmax/min reference in plaintext
                values = getattr(tensor, cmp)()
                indices = (tensor == values).float()

                with self.benchmark(niters=10, comp=comp, dim=None) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)()

                decrypted_out = encrypted_out.get_plain_text()
                self.assertTrue(decrypted_out.sum() == 1)
                self.assertTrue(decrypted_out.mul(indices).sum() == 1)

                for dim in range(tensor.dim()):

                    # Compute one-hot argmax/min reference in plaintext
                    values = getattr(tensor, cmp)(dim=dim)[0]
                    values = values.unsqueeze(dim)
                    indices = (tensor == values).float()

                    with self.benchmark(niters=10, comp=comp, dim=dim) as bench:
                        for _ in bench.iters:
                            encrypted_out = getattr(encrypted_tensor, comp)(dim=dim)
                    decrypted_out = encrypted_out.get_plain_text()
                    self.assertTrue((decrypted_out.sum(dim=dim) == 1).all())
                    self.assertTrue(
                        (decrypted_out.mul(indices).sum(dim=dim) == 1).all()
                    )

    def test_get_set(self):
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = get_random_test_tensor(size=(size, size), is_float=False)
                reference = tensor[:, 0]

                encrypted_tensor = BinarySharedTensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,), is_float=False)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = BinarySharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = BinarySharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_spdz_benchmark)
if __name__ == "__main__":
    TestBinary.benchmarks_enabled = True
    unittest.main()
