#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run tfe_benchmarks example in multiprocess mode:

$ python3 examples/mpc_imagenet/launcher.py --multiprocess

To run tfe_benchmarks example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_imagenet/mpc_imagenet.py \
      examples/mpc_imagenet/launcher.py

"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher
from mpc_imagenet import run_experiment


# input arguments:
parser = argparse.ArgumentParser(description="Encrypted inference of vision models")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
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
    "--tensorboard_folder",
    default="/tmp",
    type=str,
    help="folder in which tensorboard performs logging (default: /tmp)",
)
parser.add_argument(
    "--num_samples",
    default=None,
    type=int,
    help="number of samples to test on (default: all)",
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    # only worker with rank 0 will display logging information:
    level = logging.INFO
    rank = "0"
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
        rank = os.environ["RANK"]
    logging.getLogger().setLevel(level)

    tensorboard_folder = "/tmp/mpc_imagenet/" + rank
    os.makedirs(tensorboard_folder, exist_ok=True)
    run_experiment(
        args.model,
        imagenet_folder=args.imagenet_folder,
        tensorboard_folder=tensorboard_folder,
        num_samples=args.num_samples,
    )


def main():
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, _run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)


if __name__ == "__main__":
    main()
