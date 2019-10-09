#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run tfe_benchmarks example in multiprocess mode:

$ python3 examples/tfe_benchmarks/launcher.py --multiprocess

To run tfe_benchmarks example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/tfe_benchmarks/tfe_benchmarks.py \
      examples/tfe_benchmarks/launcher.py
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen TFEncrypted Benchmarks")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--network",
    default="B",
    type=str,
    help="choose from networks A, B and C (default: B)",
)
parser.add_argument(
    "--epochs", default=5, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--save-checkpoint-dir",
    default="/tmp/tfe_benchmarks",
    type=str,
    metavar="SAVE",
    help="path to the dir to save checkpoint (default: /tmp/tfe_benchmarks)",
)
parser.add_argument(
    "--save-modelbest-dir",
    default="/tmp/tfe_benchmarks_best",
    type=str,
    metavar="SAVE",
    help="path to the dir to save the best model (default: /tmp/tfe_benchmarks_best)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--lr-decay", default=0.1, type=float, help="lr decay factor")
parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)
parser.add_argument(
    "--mnist-dir",
    default=None,
    type=str,
    metavar="MNIST",
    help="path to the dir of MNIST raw data files",
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from tfe_benchmarks import run_tfe_benchmarks

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_tfe_benchmarks(
        args.network,
        args.epochs,
        args.start_epoch,
        args.batch_size,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.print_freq,
        args.resume,
        args.evaluate,
        args.seed,
        args.skip_plaintext,
        os.path.join(args.save_checkpoint_dir, os.environ.get("RANK", "")),
        os.path.join(args.save_modelbest_dir, os.environ.get("RANK", "")),
        mnist_dir=args.mnist_dir,
    )


def main(run_experiment):
    args = parser.parse_args()
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    os.makedirs(args.save_modelbest_dir, exist_ok=True)
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
