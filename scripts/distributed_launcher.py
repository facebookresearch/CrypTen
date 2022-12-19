#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import uuid
from argparse import ArgumentParser, REMAINDER


"""
Wrapper to launch MPC scripts as multiple processes.
"""


def main():
    args = parse_args()

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["WORLD_SIZE"] = str(args.world_size)

    processes = []

    # Use random file so multiple jobs can be run simultaneously
    INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())

    for rank in range(0, args.world_size):
        # each process's rank
        current_env["RANK"] = str(rank)
        current_env["RENDEZVOUS"] = INIT_METHOD

        # spawn the processes
        cmd = [args.training_script] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=process.args
            )


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "parties for MPC scripts"
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="The number of parties to launch." "Each party acts as its own process",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    main()
