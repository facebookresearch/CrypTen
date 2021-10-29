#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import crypten
import torch
from crypten.nn.privacy import DPSplitModel
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor


class TestNet(torch.nn.Module):
    def __init__(self, input_model):
        super().__init__()
        self.input_model = input_model
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        return self.softmax(self.input_model(x))


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
    # (model, size, loss_name)
    (torch.nn.Linear(100, 10), (20, 100), "BCELoss"),
    (TestMLP(100, 10), (20, 100), "BCELoss"),
    (TestMLPBN(100, 10), (20, 100), "BCELoss"),
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

        # TODO: Run multiple batches
        for model_tuple in TEST_MODELS:
            model, size, loss_name = model_tuple
            model = TestNet(model)

            loss_pt = getattr(torch.nn, loss_name)()
            loss_ct = getattr(crypten.nn, loss_name)()

            # Compute model gradients without DP
            features = get_random_test_tensor(size=size, is_float=True)
            features.requires_grad = True
            preds = model(features)

            # TODO: Write code to generate labels for other losses
            if loss_name == "BCELoss":
                labels = get_random_test_tensor(1, 0, preds.size(), is_float=False)
                labels = labels.float()
            else:
                labels = None
                raise NotImplementedError(f"Loss {loss_name} Not Supported Yet")
            loss = loss_pt(preds, labels)

            model.zero_grad()
            loss.backward()

            for noise_src in [None, 0, 1]:

                # Copy model so gradients do not overwrite original model for comparison
                model_ = copy.deepcopy(model)
                dp_model = DPSplitModel(
                    model_, loss_ct, NOISE_MAGNITUDE, FEATURE_SRC, LABEL_SRC, noise_src
                )

                dp_preds = dp_model(features)
                dp_model.compute_loss(dp_preds, labels)

                # Test zero_grad()
                dp_model.zero_grad()
                for p in dp_model.parameters():
                    self.assertIsNone(p.grad)

                # Test backward()
                dp_model.backward()

                if self.rank == FEATURE_SRC:
                    self._check_gradients_with_dp(model, dp_model, NOISE_MAGNITUDE)
