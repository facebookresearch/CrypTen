#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import time

import torch
from asset_minimisation_infra.dead_code_cleanup.dynamic_dispatch import dynamic_dispatch


def online_learner(
    sampler,
    dtype=torch.double,
    device="cpu",
    score_func=None,
    monitor_func=None,
    checkpoint_func=None,
    checkpoint_every=0,
):
    """
    Online learner that minimizes linear least squared loss.

    The `sampler` is expected to be an iterator that returns one sample at a time.
    Samples are assumed to be `dict`s with a `'context'` and a `'rewards'` field.

    The function takes a `dtype` and `device` as optional arguments. It also
    takes an optional `score_func` closure that can be used to plug in
    exploration mechanisms, an optional `monitor_func` closure that does logging,
    and an optional `checkpoint_func` that does checkpointing.
    """

    # initialize some variables:
    total_reward = 0.0

    # loop over dataset:
    idx = 0
    for sample in sampler():
        start_t = time.time()
        # unpack sample:
        assert "context" in sample and "rewards" in sample, (
            "invalid sample: %s" % sample
        )
        context = sample["context"].to(dtype=dtype, device=device)
        rewards = sample["rewards"].to(dtype=dtype, device=device)
        num_features, num_arms = context.nelement(), rewards.nelement()

        # initialization of model parameters:
        if idx == 0:

            # initialize accumulators for linear least squares:
            A_inv = torch.stack(
                [
                    torch.eye(num_features, dtype=dtype, device=device)
                    for _ in range(num_arms)
                ],
                dim=0,
            )  # inv(X^T X + I)
            b = torch.zeros(num_arms, num_features, dtype=dtype, device=device)  # X^T r

            # compute initial weights for all arms:
            weights = torch.zeros((num_arms, num_features), dtype=dtype, device=device)
            for arm in range(num_arms):
                weights[arm, :] = b[arm, :].matmul(A_inv[arm, :, :])

        # compute score of all arms:
        score = torch.matmul(weights, context.view(num_features, 1)).squeeze()

        # plug in exploration mechanism:
        if score_func is not None:
            score_func(score, A_inv, b, context)

        # select highest-scoring arm (break ties randomly), and observe reward:
        max_score = score.max()
        indices = torch.nonzero(score == max_score)
        selected_arm = random.choice(indices).item()

        reward = float(rewards[selected_arm].item() > random.random())

        # update linear least squares accumulators (using Sherman–Morrison formula):
        A_inv_context = A_inv[selected_arm, :, :].mv(context)
        numerator = torch.outer(A_inv_context, A_inv_context)
        denominator = A_inv_context.dot(context).add(1.0)
        A_inv[selected_arm, :, :].sub_(numerator.div_(denominator))
        b[selected_arm, :].add_(context.mul(reward))

        # update model weights:
        weights[selected_arm, :] = b[selected_arm, :].matmul(A_inv[selected_arm, :, :])

        # monitor learning progress:
        total_reward += reward
        iter_time = time.time() - start_t
        if monitor_func is not None:
            monitor_func(idx, reward, total_reward, iter_time)
        idx += 1

        # checkpointing:
        if checkpoint_func is not None and idx % checkpoint_every == 0:
            checkpoint_func(idx, {"A_inv": A_inv, "b": b})

    # signal monitoring closure that we are done:
    if monitor_func is not None:
        monitor_func(idx, None, None, None, finished=True)


@dynamic_dispatch
def epsilon_greedy(
    sampler,
    epsilon=0.0,
    dtype=torch.double,
    device="cpu",
    monitor_func=None,
    checkpoint_func=None,
    checkpoint_every=0,
):
    """
    Run epsilon-greedy linear least squares learner on dataset.

    The `sampler` is expected to be an iterator that returns one sample at a time.
    Samples are assumed to be `dict`s with a `'context'` and a `'rewards'` field.

    The function takes a hyperpameter `epsilon`, `dtype`, and `device` as optional
    arguments. It also takes an optional `monitor_func` closure that does logging,
    and an optional `checkpoint_func` that does checkpointing.
    """

    # define scoring function:
    def score_func(scores, A_inv, b, context):
        # Implement as (p < epsilon) * scores + (p > epsilon) * random
        # in order to match private version
        explore = random.random() < epsilon
        rand_scores = torch.rand_like(scores)
        scores.mul_(1 - explore).add_(rand_scores.mul(explore))

    # run online learner:
    online_learner(
        sampler,
        dtype=dtype,
        device=device,
        score_func=score_func,
        monitor_func=monitor_func,
        checkpoint_func=checkpoint_func,
        checkpoint_every=checkpoint_every,
    )


@dynamic_dispatch
def linucb(
    sampler,
    epsilon=0.1,
    dtype=torch.double,
    device="cpu",
    monitor_func=None,
    checkpoint_func=None,
    checkpoint_every=0,
):
    """
    Run LinUCB contextual bandit learner on dataset.

    The `sampler` is expected to be an iterator that returns one sample at a time.
    Samples are assumed to be `dict`s with a `'context'` and a `'rewards'` field.

    The function takes a hyperpameter `epsilon`, `dtype`, and `device` as optional
    arguments. It also takes an optional `monitor_func` closure that does logging,
    and an optional `checkpoint_func` that does checkpointing.

    Implementation following https://arxiv.org/pdf/1003.0146.pdf
    """

    # define UCB scoring function:
    def score_func(scores, A_inv, b, context):
        for arm in range(scores.nelement()):
            scores[arm] += (
                context.matmul(A_inv[arm, :, :]).dot(context).sqrt_().mul_(epsilon)
            )

    # run online learner:
    online_learner(
        sampler,
        dtype=dtype,
        device=device,
        score_func=score_func,
        monitor_func=monitor_func,
        checkpoint_func=checkpoint_func,
        checkpoint_every=checkpoint_every,
    )
