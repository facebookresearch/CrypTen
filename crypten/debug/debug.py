#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pdb as pythondebugger
import sys


class MultiprocessingPdb(pythondebugger.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin") as file:
                sys.stdin = file
                pythondebugger.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def configure_logging():
    """Configures a logging template useful for debugging multiple processes."""

    level = logging.INFO
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format=(
            "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]"
            + "[%(processName)s] %(message)s"
        ),
    )


def crypten_print(*args, dst=0, **kwargs):
    """
    Prints a message to only parties whose rank is contained by `dst` kwarg (default: 0).
    """
    if isinstance(dst, int):
        dst = [dst]
    assert isinstance(
        dst, (list, tuple)
    ), "print destination must be a list or tuple of party ranks"
    import crypten.communicator as comm

    if comm.get().get_rank() in dst:
        print(*args, **kwargs)


def crypten_log(*args, level=logging.INFO, dst=0, **kwargs):
    """
    Logs a message to logger of parties whose rank is contained by `dst` kwarg (default: 0).

    Uses logging.INFO as default level.
    """
    if isinstance(dst, int):
        dst = [dst]
    assert isinstance(
        dst, (list, tuple)
    ), "log destination must be a list or tuple of party ranks"
    import crypten.communicator as comm

    if comm.get().get_rank() in dst:
        logging.log(level, *args, **kwargs)


def crypten_print_in_order(*args, **kwargs):
    """
    Calls print(*args, **kwargs) on each party in rank order to ensure each party
    can print its full message uninterrupted and the full output is deterministic
    """
    import crypten.communicator as comm

    for i in range(comm.get().get_world_size()):
        if comm.get().get_rank() == i:
            print(*args, **kwargs)
        comm.get().barrier()
