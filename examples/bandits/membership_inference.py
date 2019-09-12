#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle

import examples.util
import experiment
import torch
import visdom


def compute_rewards(weights, dataset, epsilon=0.0):
    """
    Perform inference using epsilon-greedy contextual bandit (without updates).
    """
    context, rewards = dataset
    context = context.type(torch.float32)

    # compute scores:
    scores = torch.matmul(weights, context.t()).squeeze()
    explore = (torch.rand(scores.shape[1]) < epsilon).type(torch.float32)
    rand_scores = torch.rand_like(scores)
    scores.mul_(1 - explore).add_(rand_scores.mul(explore))

    # select arm and observe reward:
    selected_arms = scores.argmax(dim=0)
    return rewards[range(rewards.shape[0]), selected_arms]


def membership_accuracy(model, positive_set, negative_set, epsilon=0.0):
    """
    Measure accuracy of membership inference attacks on model using the specified
    positive and negative data sets.
    """

    # compute weights for all arms:
    weights = model["b"].unsqueeze(1).matmul(model["A_inv"]).squeeze(1)
    weights = weights.type(torch.float32)

    # compute rewards for both sets:
    rewards = {
        "positive": compute_rewards(weights, positive_set, epsilon=epsilon),
        "negative": compute_rewards(weights, negative_set, epsilon=epsilon),
    }

    def p_reward(x):
        return torch.sum(x).type(torch.float32) / x.numel()

    p_reward_pos = p_reward(rewards["positive"])
    p_reward_neg = p_reward(rewards["negative"])
    advantage = (p_reward_pos - p_reward_neg).abs().item()
    return advantage


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="Perform membership inference attacks")
    parser.add_argument(
        "--pca", default=20, type=int, help="Number of PCA dimensions (0 for raw data)"
    )
    parser.add_argument(
        "--number_arms",
        default=None,
        type=int,
        help="create arbitrary number of arms via k-means",
    )
    parser.add_argument(
        "--bandwidth",
        default=1.0,
        type=float,
        help="bandwidth of kernel used to assign rewards",
    )
    parser.add_argument(
        "--checkpoint_folder",
        default=None,
        type=str,
        help="folder from which to load checkpointed models",
    )
    parser.add_argument(
        "--permfile", default=None, type=str, help="file with sampling permutation"
    )
    parser.add_argument(
        "--epsilon",
        default=0.01,
        type=float,
        help="exploration parameter (default = 0.01)",
    )
    parser.add_argument(
        "--savefile", default=None, type=str, help="file to pickle advantages"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="visualize results with visdom"
    )
    return parser.parse_args()


def main():

    # parse command-line arguments:
    args = parse_args()

    # load clusters:
    clusters = None
    if args.number_arms is not None:
        clusters_file = "clusters_K=%d_pca=%d.torch" % (args.number_arms, args.pca)
        clusters_file = os.path.join(experiment.MEMOIZE_FOLDER, clusters_file)
        logging.info("Loading clusters from file...")
        clusters = torch.load(clusters_file)

    # load dataset:
    train_data, _ = experiment.load_data(split="train")
    components = examples.util.pca(train_data, args.pca)
    positive_set = experiment.load_data(
        split="train", pca=components, clusters=clusters, bandwidth=args.bandwidth
    )
    negative_set = experiment.load_data(
        split="test", pca=components, clusters=clusters, bandwidth=args.bandwidth
    )

    # get list of checkpoints:
    model_files = [
        os.path.join(args.checkpoint_folder, filename)
        for filename in os.listdir(args.checkpoint_folder)
        if filename.endswith(".torch")
    ]
    model_files = sorted(model_files)
    iterations = [int(os.path.splitext(f)[0].split("_")[-1]) for f in model_files]

    # load permutation used in training:
    perm = torch.load(args.permfile)

    def subset(dataset, iteration):
        ids = perm[:iteration]
        return tuple(d[ids, :] for d in dataset)

    # measure accuracies of membership inference attacs:
    advantage = [
        membership_accuracy(
            torch.load(model_file),
            subset(positive_set, iteration),
            negative_set,
            epsilon=args.epsilon,
        )
        for model_file, iteration in zip(model_files, iterations)
    ]

    # save advantages to file:
    if args.savefile is not None:
        with open(args.savefile, "wb") as fid:
            pickle.dump(advantage, fid)

    # plot advantages:
    if args.visualize:
        opts = {
            "xlabel": "Number of iterations",
            "ylabel": "Accuracy of inference attack",
        }
        visdom.line(iterations, advantage, opts=opts)


if __name__ == "__main__":
    main()
