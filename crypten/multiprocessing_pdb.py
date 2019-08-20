#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
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


pdb = MultiprocessingPdb()
