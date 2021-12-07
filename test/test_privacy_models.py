#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools

import crypten
import torch
from crypten.config import cfg
from crypten.nn.privacy import DPSplitModel
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor


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
    # (model, size)
    (torch.nn.Linear(100, 10), (150, 100)),
    (torch.nn.Linear(50, 5), (20, 50)),
    (torch.nn.Linear(30, 1), (50, 30)),
    (torch.nn.Linear(1, 10), (30, 1)),
    (torch.nn.Linear(1, 1), (20, 1)),
    (TestMLP(100, 10), (20, 100)),
    (TestMLPBN(100, 10), (20, 100)),
]


class TestPrivacyModels(MultiProcessTestCase):
    def _check_gradients_with_dp(self, model, dp_model, std, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)

        grad = torch.nn.utils.parameters_to_vector(model.parameters())
        dp_grad = torch.nn.utils.parameters_to_vector(dp_model.parameters())

        if std == 0:
            self.assertTrue(
                torch.allclose(grad, dp_grad, rtol=tolerance, atol=tolerance * 0.1)
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

        PROTOCOLS = ["full_jacobian", "layer_estimation"]

        # TODO: Run multiple batches
        for model_tuple, protocol in itertools.product(TEST_MODELS, PROTOCOLS):
            cfg.nn.dpsmpc.protocol = protocol

            # TODO: ensure this works with other rr_prob values
            for rr_prob in [None, 0.00001]:
                model, size = model_tuple

                # TODO test multiclass using CrossEntropyLoss()
                loss_pt = torch.nn.BCEWithLogitsLoss()

                # Compute model gradients without DP
                features = get_random_test_tensor(size=size, is_float=True)
                features.requires_grad = True

                # Get reference logits from plaintext model
                logits = model(features)

                # TODO: Write code to generate labels for CrossEntropyLoss
                labels = get_random_test_tensor(1, 0, logits.size(), is_float=False)
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
                        NOISE_MAGNITUDE,
                        FEATURE_SRC,
                        LABEL_SRC,
                        noise_src=noise_src,
                        randomized_response_prob=rr_prob,
                    )

                    dp_logits = dp_model(features)

                    # Check forward pass
                    if self.rank == FEATURE_SRC:
                        self.assertTrue(
                            dp_logits.eq(logits).all(), "model outputs do not match"
                        )

                    dp_model.compute_loss(labels_enc)

                    # Test zero_grad()
                    dp_model.zero_grad()
                    for p in dp_model.parameters():
                        self.assertIsNone(p.grad)

                    # Test backward()
                    dp_model.backward()

                    if hasattr(dp_model, "dLdW") and self.rank == FEATURE_SRC:
                        crypten.debug.pdb.set_trace()

                    if self.rank == FEATURE_SRC:
                        self._check_gradients_with_dp(model, dp_model, NOISE_MAGNITUDE)
