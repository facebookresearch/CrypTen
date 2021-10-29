#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn


class DPSplitModel(nn.Module):
    """
    Implements DP Split MPC model.

    Args:
        pytorch_model (torch.nn.Module) : The input model to be trained
            using DP-Split-MPC algorithm. Remains in plaintext throughout.
        crypten_loss (crypten.nn._Loss) : The loss function to optimize
            during model training. This loss function will be computed under
            MPC encryption.
        noise_magnitude (float) : The magnitude of DP noise to be applied to
            gradients prior to decryption for each batch of training.
        feature_src (int) : Source for input features to the model (also owns
            the plaintext model throughout training)
        label_src (int) : Source for training labels. Labels can either be input
            as plaintext values from the label_src party or as CrypTensors.
        skip_loss_forward (bool) : Determines whether we should compute the
            value of the loss during training (see crypten.nn._Loss definition
            of skip_forward). If True, this model will output zeros for the value
            of the loss function. However, correct gradients will still be computed
            when calling backward(). Default: True
        cache_pred_size (bool) :  Determines whether the size of the predictions should
            be cached. If True, DPSplitModel instances will remember the tensor and
            batch sizes input. This saves one communication round per batch, but
            the user will be responsible for using correct batch sizes to avoid
            crashing.

    Example:
        ```
        preds = dp_split_model(x)
        loss = dp_split_model.compute_loss(preds, targets)
        dp_split_model.backward()
        ```
    """

    def __init__(
        self,
        pytorch_model,
        crypten_loss,
        feature_src,
        label_src,
        noise_magnitude,
        noise_src=None,
        skip_loss_forward=True,
        cache_pred_size=True,
        randomized_response_prob=None,
    ):
        assert isinstance(
            pytorch_model, torch.nn.Module
        ), "pytorch_model must be a torch Module"
        assert isinstance(
            crypten_loss, crypten.nn._Loss
        ), "crypten_loss must be a CrypTen loss"

        super().__init__()
        self.model = pytorch_model
        self.loss_fn = crypten_loss
        self.noise_magnitude = noise_magnitude
        self.feature_src = feature_src
        self.label_src = label_src
        self.noise_src = noise_src
        self.cache_pred_size = cache_pred_size

        if randomized_response_prob is not None:
            assert (
                0 < randomized_response_prob < 0.5
            ), "randomized_response_prob must be in the interval [0, 0.5)"
        self.rr_prob = randomized_response_prob

        # Cache predictions size
        self.preds_size = None

        # Set skip_forward in crypten loss function
        self.loss_fn.skip_forward = skip_loss_forward

    def eval(self):
        self.train(mode=False)

    @property
    def training(self):
        if hasattr(self, "model"):
            return self.model.training
        return None

    @training.setter
    def training(self, mode):
        if hasattr(self, "model"):
            self.train(mode)

    def train(self, mode=True):
        self.model.train(mode=mode)
        self.loss_fn.train(mode=mode)

    def zero_grad(self):
        self.model.zero_grad()

    def forward(self, input):
        if comm.get().get_rank() == self.feature_src:
            self.preds = self.model(input)

            # Check that prediction size matches cached size
            if self.cache_pred_size and self.preds_size is not None:
                ps = self.preds.size()
                cs = self.preds_size
                if ps != cs:
                    raise ValueError(
                        f"Prediction size does not match cached size: {ps} vs. {cs}"
                    )
        else:
            self.preds = None

        # Cache predictions size - Note batch size must match here
        # TODO: Handle batch dimension here
        if self.preds_size is None:
            preds_size = self.preds.size() if self.preds is not None else None
            self.preds_size = comm.get().broadcast_obj(preds_size, self.feature_src)

        self.preds = torch.empty(self.preds_size) if self.preds is None else self.preds

        return self.preds

    @property
    def rank(self):
        return comm.get().get_rank()

    def compute_loss(self, preds, targets):
        self.preds_enc = crypten.cryptensor(
            self.preds, src=self.feature_src, requires_grad=True
        )

        # Apply appropriate RR-protocol and encrypt targets if necessary
        if self.rr_prob is not None:
            flip_probs = torch.tensor(self.rr_prob).expand(targets.size())

        if crypten.is_encrypted_tensor(targets):
            targets_enc = targets
            if self.rr_prob is not None:
                flip_mask = crypten.bernoulli(flip_probs)
                targets_enc += flip_probs - 2 * targets * flip_mask
        else:
            # Flip targets based on Randomized Response algorithm
            if self.rr_prob is not None and self.rank == self.label_src:
                flip_mask = flip_probs.bernoulli()
                targets += flip_mask - 2 * targets * flip_mask

            # Encrypt targets:
            targets_enc = crypten.cryptensor(targets, src=self.label_src)

        self.loss = self.loss_fn(self.preds_enc, targets_enc)
        return self.loss

    # TODO: Implement DP properly to make correct DP guarantees
    # TODO: Implement custom DP mechanism (split noise / magnitude)
    def _generate_noise_no_src(self, size):
        return crypten.randn(size) * self.noise_magnitude

    def _generate_noise_from_src(self, size):
        noise = torch.randn(size) * self.noise_magnitude
        noise = crypten.cryptensor(noise, src=self.noise_src)
        return noise

    # TODO: Implement de-aggregation
    def _get_noisy_dLdP(self, dLdP_enc):
        """Generates noisy dLdP using MLE de-aggregation trick"""
        raise NotImplementedError("MLE de-aggregation is not implemented.")

    def _compute_model_jacobians(self):
        """Compute Jacobians with respect to each model parameter"""
        P = self.preds.split(1, dim=-1)

        # Store partial Jacobian for each parameter
        jacobians = {}

        # dL/dW_i = sum_j (dL/dP_j * dP_j/dW_i)
        with crypten.no_grad():
            # TODO: Async / parallelize this
            for p in P:
                p.backward(torch.ones(p.size()), retain_graph=True)

                for param in self.model.parameters():
                    grad = param.grad.flatten().unsqueeze(-1)

                    # Accumulate partial gradients: dL/dP_j * dP_j/dW_i
                    if param in jacobians.keys():
                        jacobians[param] = torch.cat([jacobians[param], grad], dim=-1)
                    else:
                        jacobians[param] = grad
                    param.grad = None  # Reset grad for next p_j.backward()
        return jacobians

    def backward(self, grad_output=None):
        """Computes backward for non-RR variant.

        To add DP noise at the aggregated gradient level,
        we compute the jacobians for dP/dW in plaintext
        so we can matrix multiply by dL/dP to compute our
        gradients without performing a full backward pass in
        crypten.
        """
        # Compute dL/dP_j
        self.loss.backward(grad_output)
        dLdP = self.preds_enc.grad

        # Turn batched vector into batched matrix for matmul
        dLdP = dLdP.unsqueeze(-1)

        # Compute Jacobians wrt model weights
        if self.rank == self.feature_src:
            jacobians = self._compute_model_jacobians()

        # Populate parameter grad fields using Jacobians
        params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        if self.rank == self.feature_src:
            jacobian = torch.cat(
                [jacobians[param] for param in self.model.parameters()], dim=0
            )
        else:
            jacobian_size = (params.numel(), dLdP.size(-2))
            jacobian = torch.empty(jacobian_size)

        jacobian = crypten.cryptensor(jacobian, src=self.feature_src)

        # Compute gradeints wrt each param
        while jacobian.dim() < dLdP.dim():
            jacobian = jacobian.unsqueeze(0)
        grad = jacobian.matmul(dLdP)
        grad = grad.view(-1, *(params.size()))

        # Compute DP noise
        if not self.rr_prob:
            # Determine noise generation function
            generate_noise = (
                self._generate_noise_from_src
                if self.noise_src
                else self._generate_noise_no_src
            )
            noise = generate_noise(params.size())
            grad += noise

        # Sum over batch dimension
        while grad.size() != params.size():
            grad = grad.sum(0)

        # Decrypt dL/dP_j * dP_j/dW_i with Differential Privacy
        grads = grad.flatten().get_plain_text(dst=self.feature_src)

        # Populate grad fields of parameters:
        if self.rank == self.feature_src:
            ind = 0
            for param in self.model.parameters():
                numel = param.numel()
                param.grad = grads[ind : ind + numel].view(param.size())
                ind += numel
