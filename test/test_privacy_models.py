#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging

import crypten
import torch
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.nn.privacy import DPSplitModel, SkippedLoss
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestMLP(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, 100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.linear3 = torch.nn.Linear(50, out_features)

    def forward(self, x):
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        out = self.linear3(out)
        return out


class TestMLPBN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, 100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.linear3 = torch.nn.Linear(50, out_features)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(50)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out).relu()
        out = self.linear2(out)
        out = self.bn2(out).relu()
        out = self.linear3(out)
        return out


# TODO: Add more model types
TEST_MODELS = [
    # (model, input_size)
    (torch.nn.Linear(4, 2), (3, 4)),
    (torch.nn.Linear(100, 10), (150, 100)),
    (torch.nn.Linear(30, 1), (50, 30)),
    (torch.nn.Linear(1, 10), (30, 1)),
    # TODO: Figure out what the conditions are for input sizes - pseudo-inverse loses information
    # (torch.nn.Linear(1, 1), (5, 1)),
    (TestMLP(100, 10), (20, 100)),
    (TestMLPBN(100, 10), (20, 100)),
]

RR_PROBS = [None, 0.00001]
RAPPOR_PROBS = [None, 0.1, 0.4]


def RAPPOR_loss(alpha):
    def rappor_loss(logits, targets):
        p = logits.sigmoid()
        r = alpha * p + (1 - alpha) * (1 - p)
        return torch.nn.functional.binary_cross_entropy(r, targets)

    return rappor_loss


class TestPrivacyModels(MultiProcessTestCase):
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
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def _check_gradients_with_dp(self, model, dp_model, std, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.07)

        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        dp_grad = torch.cat([p.grad.flatten() for p in dp_model.parameters()])

        if std == 0:
            self.assertTrue(
                torch.allclose(grad, dp_grad, rtol=tolerance, atol=tolerance * 0.2)
            )
        else:
            errors = grad - dp_grad
            sample_mean = errors.mean()
            sample_std = errors.std()

            self.assertTrue(sample_mean.item() < tolerance)
            self.assertTrue(sample_std.sub(std).abs() < tolerance)

    def test_dp_split_mpc(self):
        # TODO: Vary Noise Magnitude
        NOISE_MAGNITUDE = 0
        FEATURE_SRC = 0
        LABEL_SRC = 1

        # TODO: Fix full_jacobian protocol
        # PROTOCOLS = ["full_jacobian", "layer_estimation"]
        PROTOCOLS = ["layer_estimation"]

        # TODO: Run multiple batches
        # TODO: ensure this works with other rr_prob values
        for (
            model_tuple,
            protocol,
            rr_prob,
            rappor_prob,
            skip_forward,
        ) in itertools.product(
            TEST_MODELS, PROTOCOLS, RR_PROBS, RAPPOR_PROBS, [False, True]
        ):
            logging.info(f"Model: {model_tuple}; Protocol: {protocol}")
            cfg.nn.dpsmpc.protocol = protocol
            cfg.nn.dpsmpc.skip_loss_forward = skip_forward

            model, size = model_tuple

            # TODO test multiclass using CrossEntropyLoss()
            if rappor_prob is None:
                loss_pt = torch.nn.BCEWithLogitsLoss()
            else:
                loss_pt = RAPPOR_loss(rappor_prob)

            # Compute model gradients without DP
            features = get_random_test_tensor(size=size, is_float=True)
            features.requires_grad = True

            # Get reference logits from plaintext model
            logits = model(features)

            # TODO: Write code to generate labels for CrossEntropyLoss
            labels = get_random_test_tensor(2, 0, logits.size(), is_float=False)
            labels = labels.float()
            labels_enc = crypten.cryptensor(labels, src=LABEL_SRC)

            # Compute reference loss
            loss = loss_pt(logits, labels)

            # Run reference backward pass
            model.zero_grad()
            loss.backward()

            # Delete plaintext model and features and labels for parties without access
            labels = None
            if self.rank != FEATURE_SRC:
                model = None
                features = None

            # Run split models
            for noise_src in [None, 0, 1]:
                # Copy model so gradients do not overwrite original model for comparison
                model_ = copy.deepcopy(model)
                dp_model = DPSplitModel(
                    model_,
                    FEATURE_SRC,
                    LABEL_SRC,
                    NOISE_MAGNITUDE,
                    noise_src=noise_src,
                    randomized_response_prob=rr_prob,
                    rappor_prob=rappor_prob,
                )

                dp_logits = dp_model(features)

                # Check forward pass
                if self.rank == FEATURE_SRC:
                    self.assertTrue(
                        dp_logits.eq(logits).all(), "model outputs do not match"
                    )

                dp_model.compute_loss(labels_enc)

                if skip_forward:
                    self.assertTrue(isinstance(dp_model.loss, SkippedLoss))
                else:
                    # communicate loss from feature_src party since other parties will
                    # have different losses.
                    torch.distributed.broadcast(loss, src=FEATURE_SRC)
                    self._check(
                        dp_model.loss,
                        loss,
                        "DP-Model loss is incorrect",
                        tolerance=0.15,
                    )

                # Test zero_grad()
                dp_model.zero_grad()
                for p in dp_model.parameters():
                    self.assertIsNone(p.grad)

                # Test backward()
                dp_model.backward()

                if self.rank == FEATURE_SRC:
                    self._check_gradients_with_dp(model, dp_model, NOISE_MAGNITUDE)
