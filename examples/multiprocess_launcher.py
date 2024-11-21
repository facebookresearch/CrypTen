#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import uuid

import crypten


class MultiProcessLauncher:
    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn")

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        self.processes = []
        for rank in range(world_size):
            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    world_size,
                    world_size,
                    env,
                    crypten.mpc.provider.TTPServer,
                    None,
                ),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        crypten.init()
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            assert (
                process.exitcode == 0
            ), f"{process.name} has non-zero exit code {process.exitcode}"

    def terminate(self):
        for process in self.processes:
            process.terminate()
