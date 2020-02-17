#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from crypten.mpc import primitives  # noqa: F401
from crypten.mpc import provider  # noqa: F40

from .context import run_multiprocess
from .mpc import MPCTensor
from .ptype import ptype


__all__ = ["MPCTensor", "primitives", "provider", "ptype", "run_multiprocess"]


# Set provider
__SUPPORTED_PROVIDERS = {
    "TFP": provider.TrustedFirstParty,
    "TTP": provider.TrustedThirdParty,
    "HE": provider.HomomorphicProvider,
}
__default_provider = __SUPPORTED_PROVIDERS[
    os.environ.get("CRYPTEN_PROVIDER_NAME", "TFP")
]


def set_default_provider(new_default_provider):
    global __default_provider
    assert_msg = "Provider %s is not supported" % new_default_provider
    if isinstance(new_default_provider, str):
        assert new_default_provider in __SUPPORTED_PROVIDERS.keys(), assert_msg
    else:
        assert new_default_provider in __SUPPORTED_PROVIDERS.values(), assert_msg
    __default_provider = new_default_provider
    os.environ["CRYPTEN_PROVIDER_NAME"] = new_default_provider.NAME


def get_default_provider():
    return __default_provider


def ttp_required():
    return __default_provider == provider.TrustedThirdParty
