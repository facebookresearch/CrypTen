#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import mpc_linear_svm


parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
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


def main():
    args = parser.parse_args()
    mpc_linear_svm.run_mpc_linear_svm(
        args.epochs, args.examples, args.features, args.lr, args.skip_plaintext
    )


# run all the things:
if __name__ == "__main__":
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    main()
