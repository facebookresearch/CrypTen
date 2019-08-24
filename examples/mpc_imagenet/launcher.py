#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

from mpc_imagenet import run_experiment


# input arguments:
parser = argparse.ArgumentParser(description="Encrypted inference of vision models")
parser.add_argument(
    "--model",
    default="resnet18",
    type=str,
    help="torchvision model to use for inference (default: resnet18)",
)
parser.add_argument(
    "--imagenet_folder",
    default=None,
    type=str,
    help="folder containing the ImageNet dataset",
)
parser.add_argument(
    "--num_samples",
    default=None,
    type=int,
    help="number of samples to test on (default: all)",
)


def main():
    args = parser.parse_args()
    run_experiment(
        args.model, imagenet_folder=args.imagenet_folder, num_samples=args.num_samples
    )


if __name__ == "__main__":

    # only worker with rank 0 will display logging information:
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    main()
