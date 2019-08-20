#!/usr/bin/env python3

import argparse
import logging
import os

import tfe_benchmarks


parser = argparse.ArgumentParser(description="CrypTen TFEncrypted Benchmarks")
parser.add_argument(
    "--network",
    default="B",
    type=str,
    help="choose from networks A, B and C (default: B)",
)
parser.add_argument(
    "--epochs", default=25, type=int, metavar="N", help="number of total epochs to run"
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


def main():
    args = parser.parse_args()
    tfe_benchmarks.run_tfe_benchmarks(
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
    )


if __name__ == "__main__":
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    main()
