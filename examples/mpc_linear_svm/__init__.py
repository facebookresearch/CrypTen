#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Currently just a passthrough to enable launcher.py to import
# examples.mpc_linear_svm instead of .mpc_linear_svm.
from .mpc_linear_svm import run_mpc_linear_svm

__all__ = ["run_mpc_linear_svm"]
