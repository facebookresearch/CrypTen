#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging

import crypten
import torch
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestOptim:
    """
    This class tests the crypten.optim package.
    """

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        if is_float_tensor(reference):
            diff = (tensor - reference).abs_()
            norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
            test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.2)
            test_passed = test_passed.gt(0).all().item() == 1
        else:
            test_passed = (tensor == reference).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result: %s" % tensor)
            logging.info("Reference: %s" % reference)
        self.assertTrue(test_passed, msg=msg)

    def test_sgd(self) -> None:
        lr_vals = [0.01, 0.1, 0.5]
        momentum_vals = [0.0, 0.1, 0.9]
        dampening_vals = [0.0, 0.01, 0.1]
        weight_decay_vals = [0.0, 0.9, 1.0]
        nesterov_vals = [False, True]

        torch_model = torch.nn.Linear(10, 2)
        torch_model.weight = torch.nn.Parameter(
            get_random_test_tensor(size=torch_model.weight.size(), is_float=True)
        )
        torch_model.bias = torch.nn.Parameter(
            get_random_test_tensor(size=torch_model.bias.size(), is_float=True)
        )

        crypten_model = crypten.nn.Linear(10, 2)
        crypten_model.set_parameter("weight", torch_model.weight)
        crypten_model.set_parameter("bias", torch_model.bias)
        crypten_model.encrypt()

        for lr, momentum, dampening, weight_decay, nesterov in itertools.product(
            lr_vals, momentum_vals, dampening_vals, weight_decay_vals, nesterov_vals
        ):
            kwargs = {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "dampening": dampening,
                "nesterov": nesterov,
            }

            if nesterov and (momentum <= 0 or dampening != 0):
                with self.assertRaises(ValueError):
                    crypten.optim.SGD(crypten_model.parameters(), **kwargs)
                continue

            torch_optimizer = torch.optim.SGD(torch_model.parameters(), **kwargs)
            crypten_optimizer = crypten.optim.SGD(crypten_model.parameters(), **kwargs)

            x = get_random_test_tensor(size=(10,), is_float=True)
            y = torch_model(x).sum()
            y.backward()

            xx = crypten.cryptensor(x)
            yy = crypten_model(xx).sum()
            yy.backward()

            torch_optimizer.step()
            crypten_optimizer.step()

            torch_params = list(torch_model.parameters())
            crypten_params = list(crypten_model.parameters())
            for i in range(len(torch_params)):
                self._check(
                    crypten_params[i], torch_params[i], "Parameter update mismatch"
                )

            torch_optimizer.zero_grad()
            crypten_optimizer.zero_grad()
            for i in range(len(crypten_params)):
                self.assertIsNone(crypten_params[i].grad, "Optimizer zero_grad failed")


class TestTFP(MultiProcessTestCase, TestOptim):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestOptim):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTTP, self).tearDown()
