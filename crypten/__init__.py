#!/usr/bin/env python3


def communicator():
    """Returns distributed communicator."""
    import os
    from .communicator import DistributedCommunicator

    # set default arguments for communicator:
    default_args = {
        "DISTRIBUTED_BACKEND": "gloo",
        "RENDEZVOUS": "file:///tmp/sharedfile",
        "WORLD_SIZE": 1,
        "RANK": 0,
    }
    for key, val in default_args.items():
        if key not in os.environ:
            os.environ[key] = str(val)

    # return communicator:
    return DistributedCommunicator()


# initialize communicator:
comm = communicator()

import crypten.nn  # noqa: F401

# other imports:
from .common.encrypted_tensor import EncryptedTensor
from .mpc import MPCTensor
from .multiprocessing_pdb import pdb
from .primitives import *
from .trusted_third_party import TrustedThirdParty


# expose classes and functions in package:
__all__ = ["MPCTensor", "EncryptedTensor", "TrustedThirdParty", "pdb", "nn"]
