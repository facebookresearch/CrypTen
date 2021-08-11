#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import time

import crypten
import torch
from crypten.config import cfg


def set_precision(bits):
    cfg.encoder.precision_bits = bits


def online_learner(
    sampler,
    backend="mpc",
    nr_iters=7,
    score_func=None,
    monitor_func=None,
    checkpoint_func=None,
    checkpoint_every=0,
):
    """
    Online learner that minimizes linear least squared loss.

    Args:
        sampler: An iterator that returns one sample at a time. Samples are
            assumed to be `dict`s with a `'context'` and a `'rewards'` field.
        backend: Which privacy protocol to use (default 'mpc').
        score_func: A closure that can be used to plug in exploration mechanisms.
        monitor_func: A closure that does logging.
        checkpoint_func: A closure that does checkpointing.
        nr_iters: Number of Newton-Rhapson iterations to use for private
            reciprocal.
    """

    # initialize some variables:
    total_reward = 0.0

    # initialize constructor for tensors:
    crypten.set_default_backend(backend)

    # loop over dataset:
    idx = 0
    for sample in sampler():
        start_t = time.time()

        # unpack sample:
        assert "context" in sample and "rewards" in sample, (
            "invalid sample: %s" % sample
        )

        context = crypten.cryptensor(sample["context"])
        num_features = context.nelement()
        num_arms = sample["rewards"].nelement()

        # initialization of model parameters:
        if idx == 0:

            # initialize accumulators for linear least squares:
            A_inv = [torch.eye(num_features).unsqueeze(0) for _ in range(num_arms)]
            A_inv = crypten.cat([crypten.cryptensor(A) for A in A_inv])
            b = crypten.cryptensor(torch.zeros(num_arms, num_features))

            # compute initial weights for all arms:
            weights = b.unsqueeze(1).matmul(A_inv).squeeze(1)

        # compute score of all arms:
        scores = weights.matmul(context)

        # plug in exploration mechanism:
        if score_func is not None:
            score_func(scores, A_inv, b, context)

        onehot = scores.argmax()

        # In practice only one party opens the onehot vector in order to
        # take the action.
        selected_arm = onehot.get_plain_text().argmax()

        # Once the action is taken, the reward (a scalar) is observed by some
        # party and secret shared. Here we simulate that by selecting the
        # reward from the rewards vector and then sharing it.
        reward = crypten.cryptensor(
            (sample["rewards"][selected_arm] > random.random()).view(1).float()
        )

        # update linear least squares accumulators (using Sherman–Morrison
        # formula):
        A_inv_context = A_inv.matmul(context)
        numerator = A_inv_context.unsqueeze(1).mul(A_inv_context.unsqueeze(2))
        denominator = A_inv_context.matmul(context).add(1.0).view(-1, 1, 1)
        with crypten.mpc.ConfigManager("reciprocal_nr_iters", nr_iters):
            update = numerator.mul_(denominator.reciprocal())
        A_inv.sub_(update.mul_(onehot.view(-1, 1, 1)))
        b.add_(context.mul(reward).unsqueeze(0).mul_(onehot.unsqueeze(0)))

        # update model weights:
        weights = b.unsqueeze(1).matmul(A_inv).squeeze(1)

        # monitor learning progress: we use the plain reward only for
        # monitoring
        reward = reward.get_plain_text().item()
        total_reward += reward
        iter_time = time.time() - start_t
        if monitor_func is not None:
            monitor_func(idx, reward, total_reward, iter_time)
        idx += 1

        # checkpointing:
        if checkpoint_func is not None and idx % checkpoint_every == 0:
            checkpoint_func(
                idx,
                {
                    "A_inv": [AA.get_plain_text() for AA in A_inv],
                    "b": [bb.get_plain_text() for bb in b],
                },
            )

    # signal monitoring closure that we are done:
    if monitor_func is not None:
        monitor_func(idx, None, None, None, finished=True)


def epsilon_greedy(
    sampler,
    epsilon=0.0,
    backend="mpc",
    nr_iters=7,
    precision=20,
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

    set_precision(precision)

    # define scoring function
    def score_func(scores, A_inv, b, context):
        explore = crypten.bernoulli(torch.tensor([epsilon]))
        rand_scores = crypten.rand(*scores.size())
        scores.mul_(1 - explore).add_(rand_scores.mul(explore))

    # run online learner:
    online_learner(
        sampler,
        backend=backend,
        score_func=score_func,
        monitor_func=monitor_func,
        checkpoint_func=checkpoint_func,
        checkpoint_every=checkpoint_every,
        nr_iters=nr_iters,
    )
