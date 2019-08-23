#!/usr/bin/env python3


# dependencies:
import logging
import math
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import torch


# placeholders for class references, to be filled later by import_crypten():
MPCTensor, is_float_tensor, crypten = None, None, None
ArithmeticSharedTensor, BinarySharedTensor = None, None


def import_crypten():
    """
    Imports CrypTen types. This function is called after environment variables
    in MultiProcessTestCase.setUp() are set, and sets the class references for
    all test functions.
    """
    global MPCTensor, is_float_tensor, crypten
    global ArithmeticSharedTensor, BinarySharedTensor

    from crypten import MPCTensor as _MPCTensor
    from crypten.common.tensor_types import is_float_tensor as _is_float_tensor
    import crypten as _crypten

    MPCTensor = _MPCTensor
    is_float_tensor = _is_float_tensor
    crypten = _crypten
    ArithmeticSharedTensor = crypten.primitives.arithmetic.ArithmeticSharedTensor
    BinarySharedTensor = crypten.primitives.binary.BinarySharedTensor


class TestCrypten(MultiProcessTestCase):
    """
        This class tests all member functions of crypten package
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        if self.rank >= 0:
            import_crypten()

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

    def test_cat_stack(self):
        """Tests concatenation and stacking of tensors"""
        tensor1 = get_random_test_tensor(size=(5, 5, 5, 5), is_float=True)
        tensor2 = get_random_test_tensor(size=(5, 5, 5, 5), is_float=True)

        for type1 in [lambda x: x, MPCTensor]:
            encrypted1 = type1(tensor1)
            for type2 in [lambda x: x, MPCTensor]:
                encrypted2 = type2(tensor2)

                for op in ["cat", "stack"]:
                    reference = getattr(torch, op)([tensor1, tensor2])
                    with self.benchmark(type=op) as bench:
                        for _ in bench.iters:
                            encrypted_out = getattr(crypten, op)(
                                [encrypted1, encrypted2]
                            )
                    self._check(encrypted_out, reference, "%s failed" % op)

                    for dim in range(4):
                        reference = getattr(torch, op)([tensor1, tensor2], dim=dim)
                        with self.benchmark(type=op, dim=dim) as bench:
                            for _ in bench.iters:
                                encrypted_out = getattr(crypten, op)(
                                    [encrypted1, encrypted2], dim=dim
                                )
                        self._check(encrypted_out, reference, "%s failed" % op)

    def test_rand(self):
        """Tests uniform random variable generation on [0, 1)"""
        for size in [(10,), (10, 10), (10, 10, 10)]:
            with self.benchmark(size=size) as bench:
                for _ in bench.iters:
                    randvec = crypten.rand(*size)
            self.assertTrue(randvec.size() == size, "Incorrect size")
            tensor = randvec.get_plain_text()
            self.assertTrue(
                (tensor >= 0).all() and (tensor < 1).all(), "Invalid values"
            )

        randvec = crypten.rand(int(1e6)).get_plain_text()
        mean = torch.mean(randvec)
        var = torch.var(randvec)
        self.assertTrue(torch.isclose(mean, torch.Tensor([0.5]), rtol=1e-3, atol=1e-3))
        self.assertTrue(
            torch.isclose(var, torch.Tensor([1.0 / 12]), rtol=1e-3, atol=1e-3)
        )

    def test_bernoulli(self):
        for size in [(10,), (10, 10), (10, 10, 10)]:
            probs = torch.rand(size)
            with self.benchmark(size=size) as bench:
                for _ in bench.iters:
                    randvec = crypten.bernoulli(probs)
            self.assertTrue(randvec.size() == size, "Incorrect size")
            tensor = randvec.get_plain_text()
            self.assertTrue(((tensor == 0) + (tensor == 1)).all(), "Invalid values")

        probs = torch.Tensor(int(1e6)).fill_(0.2)
        randvec = crypten.bernoulli(probs).get_plain_text()
        frac_zero = float((randvec == 0).sum()) / randvec.nelement()
        self.assertTrue(math.isclose(frac_zero, 0.8, rel_tol=1e-3, abs_tol=1e-3))

    def test_ptype(self):
        """Test that ptype attribute creates the correct type of encrypted tensor"""
        import crypten

        ptype_values = [crypten.arithmetic, crypten.binary]
        tensor_types = [ArithmeticSharedTensor, BinarySharedTensor]
        for i, curr_ptype in enumerate(ptype_values):
            tensor = get_random_test_tensor(is_float=False)
            encr_tensor = MPCTensor(tensor, ptype=curr_ptype)
            assert isinstance(encr_tensor._tensor, tensor_types[i]), "ptype test failed"


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestCrypten.benchmarks_enabled = True
    unittest.main()
