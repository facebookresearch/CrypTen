#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import torch.multiprocessing as mp
import uuid

import crypten


class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        self.spawn_context = mp.spawn(
            fn=self.__class__._run_process,
            args=(env, run_process_fn, fn_args),
            nprocs=world_size,
            join=False,
        )

    @classmethod
    def _run_process(cls, rank, env, run_process_fn, fn_args):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        crypten.init()
        logging.getLogger().setLevel(orig_logging_level)
        run_process_fn(fn_args)

    def start(self):
        pass

    def join(self):
        self.spawn_context.join()

    def terminate(self):
        pass
