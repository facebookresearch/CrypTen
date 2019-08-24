#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class NoopContextManager:
    """Context manager that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
