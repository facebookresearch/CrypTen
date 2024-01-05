#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import crypten
import torch
from crypten.config import cfg
from test.multiprocess_test_case import MultiProcessTestCase


class TestDistributions:
    """
    This class tests accuracy of distributions provided by random sampling in crypten.
    """

    def _check_distribution(
        self, func, expected_mean, expected_variance, lb=None, ub=None
    ):
        """
        Checks that the function `func` returns a distribution with the expected
        size, mean, and variance.

        Arguments:
            func - A function that takes a size and returns a random sample as a CrypTensor
            expected_mean - The expected mean for the distribution returned by function `func`
            expected_variance - The expected variance for the distribution returned by function `func
            lb - An expected lower bound on samples from the given distribution. Use None if -Inf.
            ub - An expected uppder bound on samples from the given distribution. Use None if +Inf.
        """
        name = func.__name__
        for size in [(10000,), (1000, 10), (101, 11, 11)]:
            sample = func(size)

            self.assertTrue(
                sample.size() == size, "Incorrect size for %s distribution" % name
            )

            plain_sample = sample.get_plain_text().float()
            mean = plain_sample.mean()
            var = plain_sample.var()
            self.assertTrue(
                math.isclose(mean, expected_mean, rel_tol=1e-1, abs_tol=1e-1),
                "incorrect variance for %s distribution: %f" % (name, mean),
            )
            self.assertTrue(
                math.isclose(var, expected_variance, rel_tol=1e-1, abs_tol=1e-1),
                "incorrect variance for %s distribution: %f" % (name, var),
            )
            if lb is not None:
                self.assertTrue(
                    plain_sample.ge(lb).all(),
                    "Sample detected below lower bound for %s distribution" % name,
                )
            if ub is not None:
                self.assertTrue(
                    plain_sample.le(ub).all(),
                    "Sample detected below lower bound for %s distribution" % name,
                )

    def test_uniform(self):
        self._check_distribution(crypten.rand, 0.5, 0.083333, lb=0, ub=1)

    def test_normal(self):
        self._check_distribution(crypten.randn, 0, 1)

    def test_bernoulli(self):
        for p in [0.25 * i for i in range(5)]:

            def bernoulli(*size):
                x = crypten.cryptensor(p * torch.ones(*size))
                return x.bernoulli()

            self._check_distribution(bernoulli, p, p * (1 - p), lb=0, ub=1)

            # Assert all values are in discrete set {0, 1}
            tensor = bernoulli((1000,)).get_plain_text()
            self.assertTrue(
                ((tensor == 0) + (tensor == 1)).all(), "Invalid Bernoulli values"
            )


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestDistributions):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestDistributions):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTTP, self).tearDown()
