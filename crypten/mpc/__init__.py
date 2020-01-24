#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from crypten.mpc import primitives  # noqa: F401
from crypten.mpc import provider  # noqa: F40

from .context import run_multiprocess
from .mpc import MPCTensor
from .ptype import ptype


__all__ = ["MPCTensor", "primitives", "provider", "ptype", "run_multiprocess"]


def __cat_stack_helper(op, tensors, *args, **kwargs):
    assert op in ["cat", "stack"], "Unsupported op for helper function"
    assert isinstance(tensors, list), "%s input must be a list" % op
    assert len(tensors) > 0, "expected a non-empty list of MPCTensors"

    _ptype = kwargs.pop("ptype", None)
    # Populate ptype field
    if _ptype is None:
        for tensor in tensors:
            if isinstance(tensor, MPCTensor):
                _ptype = tensor.ptype
                break
    if _ptype is None:
        _ptype = ptype.arithmetic

    # Make all inputs MPCTensors of given ptype
    for i, tensor in enumerate(tensors):
        if torch.is_tensor(tensor):
            tensors[i] = MPCTensor(tensor, ptype=_ptype)
        assert isinstance(tensors[i], MPCTensor), "Can't %s %s with MPCTensor" % (
            op,
            type(tensor),
        )
        if tensors[i].ptype != _ptype:
            tensors[i] = tensors[i].to(_ptype)

    # Operate on all input tensors
    result = tensors[0].clone()
    result.share = getattr(torch, op)(
        [tensor.share for tensor in tensors], *args, **kwargs
    )
    return result


def cat(tensors, *args, **kwargs):
    """Perform matrix concatenation"""
    return __cat_stack_helper("cat", tensors, *args, **kwargs)


def stack(tensors, *args, **kwargs):
    """Perform tensor stacking"""
    return __cat_stack_helper("stack", tensors, *args, **kwargs)


def rand(*sizes):
    """
    Returns a tensor with elements uniformly sampled in [0, 1) using the
    trusted third party.
    """
    rand = MPCTensor(None)
    rand._tensor = __default_provider.rand(*sizes)
    rand.ptype = ptype.arithmetic
    return rand


def bernoulli(tensor):
    """
    Returns a tensor with elements in {0, 1}. The i-th element of the
    output will be 1 with probability according to the i-th value of the
    input tensor.
    """
    return rand(tensor.size()) < tensor


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
