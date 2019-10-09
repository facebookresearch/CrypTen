#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_linear_svm example in multiprocess mode:

$ python3 examples/mpc_linear_svm/launcher.py --multiprocess

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_linear_svm/mpc_linear_svm.py \
      examples/mpc_linear_svm/launcher.py
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen Linear SVM Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--examples", default=50, type=int, metavar="N", help="number of examples per epoch"
)
parser.add_argument(
    "--features",
    default=100,
    type=int,
    metavar="N",
    help="number of features per example",
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.5, type=float, help="initial learning rate"
)
parser.add_argument(
    "--skip_plaintext",
    default=False,
    action="store_true",
    help="skip evaluation for plaintext svm",
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from mpc_linear_svm import run_mpc_linear_svm

    run_mpc_linear_svm(
        args.epochs, args.examples, args.features, args.lr, args.skip_plaintext
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
