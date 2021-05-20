#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .debug import MultiprocessingPdb, configure_logging, validate_correctness


pdb = MultiprocessingPdb()

__all__ = ["pdb", "configure_logging", "validate_correctness"]


# debug mode handling
_crypten_debug_mode = False
_crypten_validation_mode = False


def debug_mode():
    return _crypten_debug_mode


def set_debug_mode(mode=True):
    assert isinstance(mode, bool)
    global _crypten_debug_mode
    _crypten_debug_mode = mode


def validation_mode():
    return _crypten_validation_mode


def set_validation_mode(mode=True):
    assert isinstance(mode, bool)
    global _crypten_validation_mode
    _crypten_validation_mode = mode
