#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .debug import MultiprocessingPdb, configure_logging


pdb = MultiprocessingPdb()

__all__ = ["pdb", "configure_logging"]


# debug mode handling
_debug_mode = False


def debug_mode():
    return _debug_mode


def set_debug_mode(mode=True):
    assert isinstance(mode, bool)
    global _debug_mode
    _debug_mode = mode
