#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import multiprocessing
import os
import tempfile
from operator import itemgetter

import crypten
from crypten.communicator import DistributedCommunicator


def _launch(func, rank, world_size, rendezvous_file, queue, func_args, func_kwargs):
    communicator_args = {
        "WORLD_SIZE": world_size,
        "RANK": rank,
        "RENDEZVOUS": "file://%s" % rendezvous_file,
        "DISTRIBUTED_BACKEND": "gloo",
    }
    for key, val in communicator_args.items():
        os.environ[key] = str(val)

    crypten.init()
    return_value = func(*func_args, **func_kwargs)
    crypten.uninit()

    queue.put((rank, return_value))


def run_multiprocess(world_size, maxsize=None):
    """Defines decorator to run function across multiple processes

    Args:
        world_size (int): number of parties / processes to initiate.
        maxsize: Enables the user to increase the size of returnable values
            (See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rendezvous_file = tempfile.NamedTemporaryFile(delete=True).name

            if maxsize is None:
                queue = multiprocessing.Queue()
            else:
                queue = multiprocessing.Queue(maxsize)

            processes = [
                multiprocessing.Process(
                    target=_launch,
                    args=(func, rank, world_size, rendezvous_file, queue, args, kwargs),
                )
                for rank in range(world_size)
            ]

            # Initialize TTP process
            if crypten.mpc.ttp_required():
                processes += [
                    multiprocessing.Process(
                        target=_launch,
                        args=(
                            crypten.mpc.provider.TTPServer,
                            world_size,
                            world_size,
                            rendezvous_file,
                            queue,
                            (),
                            {},
                        ),
                    )
                ]

            # This process will be forked and we need to re-initialize the
            # communicator in the children. If the parent process happened to
            # call crypten.init(), which might be valid in a Jupyter notebook
            # for instance, then the crypten.init() call on the children
            # process will not do anything. The call to uninit here makes sure
            # we actually get to initialize the communicator on the child
            # process.  An alternative fix for this issue would be to use spawn
            # instead of fork, but we run into issues serializing the function
            # in that case.
            was_initialized = DistributedCommunicator.is_initialized()
            if was_initialized:
                crypten.uninit()

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            if was_initialized:
                crypten.init()

            successful = [process.exitcode == 0 for process in processes]
            if not all(successful):
                logging.error("One of the parties failed. Check past logs")
                return None

            return_values = []
            while not queue.empty():
                return_values.append(queue.get())

            return [value for _, value in sorted(return_values, key=itemgetter(0))]

        return wrapper

    return decorator
