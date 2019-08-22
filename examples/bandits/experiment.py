#!/usr/bin/env python3

import argparse
import logging
import os
import random

import torch
import util
import visdom
from torchvision.datasets.mnist import MNIST


def learning_curve(visualizer, idx, value, window=None, title=""):
    """
    Appends new value to learning curve, creating new curve if none exists.
    """
    opts = {"title": title, "xlabel": "Number of samples", "ylabel": "Reward value"}
    window = visualizer.line(
        value.view(value.nelement(), 1),
        idx,
        update=None if window is None else "append",
        opts=opts,
        win=window,
        env="contextual_bandits",
    )
    return window


def load_data(split="train", pca=None, clusters=None, bandwidth=1.0):
    """
    Loads split from the MNIST dataset and returns data.
    """

    # download the MNIST dataset:
    with util.PublicDownloader():
        mnist = MNIST("/tmp", download=True, train=(split == "train"))

    # preprocess the MNIST dataset:
    context = mnist.data.float().div_(255.0)
    context = context.view(context.size(0), -1)

    # apply PCA:
    if pca is not None:
        context -= torch.mean(context, dim=0, keepdim=True)
        context = context.matmul(pca)
        context /= torch.norm(context, dim=1, keepdim=True)

    # compute rewards (based on clustering if clusters defined, 0-1 otherwise):
    if clusters is not None:
        assert clusters.size(1) == context.size(
            1
        ), "cluster dimensionality does not match data dimensionality"
        rewards = util.kmeans_inference(
            context, clusters, hard=False, bandwidth=bandwidth
        )
    else:
        rewards = util.onehot(mnist.targets.long())

    # return data:
    return context, rewards


def load_data_sampler(
    split="train", pca=None, clusters=None, bandwidth=1.0, permfile=None
):
    """
    Loads split from the MNIST dataset and returns sampler.
    """

    # load dataset:
    context, rewards = load_data(
        split=split, pca=pca, clusters=clusters, bandwidth=bandwidth
    )
    if permfile is not None:
        perm = torch.load(permfile)
        assert perm.shape[0] == context.shape[0], "Incorrect perm size for context."
    else:
        perm = torch.randperm(context.size(0))

    # define simple dataset sampler:
    def sampler():
        idx = 0
        while idx < context.size(0):
            yield {"context": context[perm[idx], :], "rewards": rewards[perm[idx], :]}
            idx += 1

    # return sampler:
    return sampler


def parse_args(hostname):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train contextual bandit model using encrypted learning signal"
    )
    parser.add_argument(
        "--plaintext", action="store_true", help="use a non-private algorithm"
    )
    parser.add_argument(
        "--backend", default="mpc", type=str, help="crypten backend: mpc (default)"
    )
    parser.add_argument(
        "--learner",
        default="epsilon_greedy",
        type=str,
        help="learning algorithm: epsilon_greedy or linucb",
    )
    parser.add_argument(
        "--epsilon",
        default=0.01,
        type=float,
        help="exploration parameter (default = 0.01)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="visualize results with visdom"
    )
    parser.add_argument(
        "--visdom",
        default=hostname,
        type=str,
        help="visdom server to use (default = %s)" % hostname,
    )
    parser.add_argument(
        "--pca", default=20, type=int, help="Number of PCA dimensions (0 for raw data)"
    )
    parser.add_argument(
        "--precision",
        default=20,
        type=int,
        help="Bits of precision for encoding floats.",
    )
    parser.add_argument(
        "--nr_iters",
        default=7,
        type=int,
        help="Newton-Rhapson iterations for mpc reciprocal",
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
        "--memoize_folder",
        default="/tmp/kmeans",
        type=str,
        help="folder to save k-means clusters",
    )
    parser.add_argument(
        "--checkpoint_folder",
        default=None,
        type=str,
        help="folder in which to checkpoint models",
    )
    parser.add_argument(
        "--checkpoint_every",
        default=1000,
        type=int,
        help="checkpoint every K iterations",
    )
    parser.add_argument(
        "--permfile", default=None, type=str, help="file with sampling permutation"
    )
    parser.add_argument("--seed", default=None, type=int, help="Seed the torch rng")
    return parser.parse_args()


def get_monitor_func(args, buffers, visualizer, window, title, progress_iter):
    """
    Return closure that performs monitoring.
    """

    def monitor_func(idx, reward, total_reward, iter_time, finished=False):
        def mean(vals):
            return torch.DoubleTensor(vals).mean().item()

        # flush buffers:
        if finished:
            for key, val in buffers.items():
                buffers[key] = [item for item in val if item is not None]
        if finished or (idx > 0 and idx % progress_iter == 0):
            logging.info(
                "Sample %s; average reward = %2.5f, time %.3f (sec/iter) "
                % (idx, mean(buffers["reward"]), mean(buffers["iter_time"]))
            )
            if args.visualize:
                window[0] = learning_curve(
                    visualizer,
                    torch.LongTensor(buffers["idx"]),
                    torch.DoubleTensor(buffers["cumulative_reward"]),
                    window=window[0],
                    title=title,
                )
            for key in buffers.keys():
                buffers[key] = [None] * progress_iter

        # fill buffers:
        if idx is not None:
            cur_idx = idx % progress_iter
            buffers["idx"][cur_idx] = idx
            buffers["reward"][cur_idx] = reward
            buffers["cumulative_reward"][cur_idx] = total_reward
            buffers["iter_time"][cur_idx] = iter_time

    return monitor_func


def get_checkpoint_func(args):
    """
    Return closure that performs checkpointing.
    """

    def checkpoint_func(idx, model):
        if "RANK" not in os.environ or os.environ["RANK"] == 0:
            if args.checkpoint_folder is not None:
                checkpoint_file = os.path.join(
                    args.checkpoint_folder, "iter_%05d.torch" % idx
                )
                torch.save(model, checkpoint_file)

    return checkpoint_func


def main():
    """
    Runs encrypted contextual bandits learning experiment on MNIST.
    """

    # parse input arguments:
    args = parse_args(os.environ.get("HOSTNAME", "localhost"))

    # set up loggers:
    logger = logging.getLogger()
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logger.setLevel(level)
    visualizer = visdom.Visdom(args.visdom) if args.visualize else None

    # allow comparisons between plain and private algorithm:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.plaintext:
        logging.info("Using plain text bandit")
        kwargs = {"dtype": torch.double, "device": "cpu"}
        import plain_contextual_bandits as bandits
    else:
        logging.info(f"Using encrypted bandit with {args.backend}")
        kwargs = {
            "backend": args.backend,
            "precision": args.precision,
            "nr_iters": args.nr_iters,
        }
        import private_contextual_bandits as bandits

    # set up variables for progress monitoring:
    window = [None]
    title = "Cumulative reward (encrypted %s, epsilon = %2.2f)" % (
        args.learner,
        args.epsilon,
    )
    progress_iter = 100
    buffers = {
        key: [None] * progress_iter
        for key in ["idx", "reward", "cumulative_reward", "iter_time"]
    }

    # closures that perform progress monitoring and checkpointing:
    monitor_func = get_monitor_func(
        args, buffers, visualizer, window, title, progress_iter
    )
    checkpoint_func = get_checkpoint_func(args)

    # compute pca:
    context, _ = load_data(split="train", pca=None)
    pca = util.pca(context, args.pca)

    # create or load clustering if custom number of arms is used:
    clusters = None
    if args.number_arms is not None:
        clusters_file = "clusters_K=%d_pca=%d.torch" % (args.number_arms, args.pca)
        clusters_file = os.path.join(args.memoize_folder, clusters_file)

        # load precomputed clusters from file:
        if os.path.exists(clusters_file):
            logging.info("Loading clusters from file...")
            clusters = torch.load(clusters_file)
        else:

            # load data and allocate clusters:
            context, _ = load_data(split="train", pca=pca)
            clusters = context.new((args.number_arms, context.size(1)))

            # run clustering in process 0:
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                logging.info("Performing clustering to get arms...")
                clusters = util.kmeans(context, args.number_arms)
                torch.save(clusters, clusters_file)

            # if run is distributed, synchronize clusters:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.broadcast(clusters, 0)

    # run contextual bandit algorithm on MNIST:
    sampler = load_data_sampler(
        pca=pca, clusters=clusters, bandwidth=args.bandwidth, permfile=args.permfile
    )
    assert hasattr(bandits, args.learner), "unknown learner: %s" % args.learner
    getattr(bandits, args.learner)(
        sampler,
        epsilon=args.epsilon,
        monitor_func=monitor_func,
        checkpoint_func=checkpoint_func,
        checkpoint_every=args.checkpoint_every,
        **kwargs,
    )


# run all the things:
if __name__ == "__main__":
    main()
