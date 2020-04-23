#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.1.0"

import copy
import warnings

import crypten.communicator as comm
import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401
import torch

# other imports:
from . import debug
from .cryptensor import CrypTensor


# functions controlling autograd:
no_grad = CrypTensor.no_grad
enable_grad = CrypTensor.enable_grad
set_grad_enabled = CrypTensor.set_grad_enabled


def init():
    comm._init(use_threads=False, init_ttp=crypten.mpc.ttp_required())
    if comm.get().get_rank() < comm.get().get_world_size():
        _setup_przs()
        if crypten.mpc.ttp_required():
            crypten.mpc.provider.ttp_provider.TTPClient._init()


def init_thread(rank, world_size):
    comm._init(use_threads=True, rank=rank, world_size=world_size)
    _setup_przs()


def uninit():
    return comm.uninit()


def is_initialized():
    return comm.is_initialized()


def print_communication_stats():
    comm.get().print_communication_stats()


def reset_communication_stats():
    comm.get().reset_communication_stats()


# set tensor type to be used for CrypTensors:
__CRYPTENSOR_TYPES__ = {"mpc": crypten.mpc.MPCTensor}
__DEFAULT_CRYPTENSOR_TYPE__ = "mpc"


def register_cryptensor(name):
    """Registers a custom :class:`CrypTensor` subclass.

    This decorator allows the user to instantiate a subclass of `CrypTensor`
    from Python cpde, even if the class itself is not  part of CrypTen. To use
    it, apply this decorator to a `CrypTensor` subclass, like this:

    .. code-block:: python

        @crypten.register_cryptensor('my_cryptensor')
        class MyCrypTensor(crypten.CrypTensor):
            ...
    """

    def register_cryptensor_cls(cls):
        if name in __CRYPTENSOR_TYPES__:
            raise ValueError(
                "Cannot register duplicate CrypTensor type: \
                tensor type {} already exists.".format(
                    name
                )
            )
        if not issubclass(cls, CrypTensor):
            raise ValueError(
                "Registered tensor ({}: {}) must extend \
                CrypTensor".format(
                    name, cls.__name__
                )
            )
        __CRYPTENSOR_TYPES__[name] = cls
        return cls

    return register_cryptensor_cls


def set_default_cryptensor_type(cryptensor_type):
    """Sets the default type used to create `CrypTensor`s."""
    global __DEFAULT_CRYPTENSOR_TYPE__
    if cryptensor_type not in __CRYPTENSOR_TYPES__:
        raise ValueError("CrypTensor type %s does not exist." % cryptensor_type)
    __DEFAULT_CRYPTENSOR_TYPE__ = cryptensor_type


def get_default_cryptensor_type():
    """Gets the default type used to create `CrypTensor`s."""
    return __DEFAULT_CRYPTENSOR_TYPE__


def get_cryptensor_type(tensor):
    """Gets the type name of the specified `tensor` `CrypTensor`."""
    if not isinstance(tensor, CrypTensor):
        raise ValueError(
            "Specified tensor is not a CrypTensor: {}".format(type(tensor))
        )
    for name, cls in __CRYPTENSOR_TYPES__.items():
        if isinstance(tensor, cls):
            return name
    raise ValueError("Unregistered CrypTensor type: {}".format(type(tensor)))


def cryptensor(*args, cryptensor_type=None, **kwargs):
    """
    Factory function to return encrypted tensor of given `cryptensor_type`. If no
    `cryptensor_type` is specified, the default type is used.
    """

    # determine CrypTensor type to use:
    if cryptensor_type is None:
        cryptensor_type = get_default_cryptensor_type()
    if cryptensor_type not in __CRYPTENSOR_TYPES__:
        raise ValueError("CrypTensor type %s does not exist." % cryptensor_type)

    # create CrypTensor:
    return __CRYPTENSOR_TYPES__[cryptensor_type](*args, **kwargs)


def is_encrypted_tensor(obj):
    """
    Returns True if obj is an encrypted tensor.
    """
    return isinstance(obj, CrypTensor)


def _setup_przs():
    """
        Generate shared random seeds to generate pseudo-random sharings of
        zero. The random seeds are shared such that each process shares
        one seed with the previous rank process and one with the next rank.
        This allows for the generation of `n` random values, each known to
        exactly two of the `n` parties.

        For arithmetic sharing, one of these parties will add the number
        while the other subtracts it, allowing for the generation of a
        pseudo-random sharing of zero. (This can be done for binary
        sharing using bitwise-xor rather than addition / subtraction)
    """
    # Initialize RNG Generators
    comm.get().g0 = torch.Generator()
    comm.get().g1 = torch.Generator()

    # Generate random seeds for Generators
    # NOTE: Chosen seed can be any number, but we choose as a random 64-bit
    # integer here so other parties cannot guess its value.

    # We sometimes get here from a forked process, which causes all parties
    # to have the same RNG state. Reset the seed to make sure RNG streams
    # are different in all the parties. We use numpy's random here since
    # setting its seed to None will produce different seeds even from
    # forked processes.
    import numpy

    numpy.random.seed(seed=None)
    next_seed = torch.tensor(numpy.random.randint(-2 ** 63, 2 ** 63 - 1, (1,)))
    prev_seed = torch.LongTensor([0])  # placeholder

    # Send random seed to next party, receive random seed from prev party
    world_size = comm.get().get_world_size()
    rank = comm.get().get_rank()
    if world_size >= 2:  # Otherwise sending seeds will segfault.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        req0 = comm.get().isend(tensor=next_seed, dst=next_rank)
        req1 = comm.get().irecv(tensor=prev_seed, src=prev_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed

    # Seed Generators
    comm.get().g0.manual_seed(next_seed.item())
    comm.get().g1.manual_seed(prev_seed.item())

    # Create global generator
    global_seed = torch.tensor(numpy.random.randint(-2 ** 63, 2 ** 63 - 1, (1,)))
    global_seed = comm.get().broadcast(global_seed, 0)
    comm.get().global_generator = torch.Generator()
    comm.get().global_generator.manual_seed(global_seed.item())


def load_from_party(
    f=None,
    preloaded=None,
    encrypted=False,
    dummy_model=None,
    src=0,
    load_closure=torch.load,
    **kwargs
):
    """
    Loads an object saved with `torch.save()` or `crypten.save_from_party()`.

    Args:
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        preloaded: Use the preloaded value instead of loading a tensor/model from f.
        encrypted: Determines whether crypten should load an encrypted tensor
                      or a plaintext torch tensor.
        dummy_model: Takes a model architecture to fill with the loaded model
                    (on the `src` party only). Non-source parties will return the
                    `dummy_model` input (with data unchanged). Loading a model will
                    assert the correctness of the model architecture provided against
                    the model loaded. This argument is ignored if the file loaded is
                    a tensor. (deprecated)
        src: Determines the source of the tensor. If `src` is None, each
            party will attempt to read in the specified file. If `src` is
            specified, the source party will read the tensor from `f` and it
            will broadcast it to the other parties
        load_closure: Custom load function that matches the interface of `torch.load`,
        to be used when the tensor is saved with a custom save function in
        `crypten.save_from_party`. Additional kwargs are passed on to the closure.
    """
    if dummy_model is not None:
        warnings.warn(
            "dummy_model is deprecated and no longer required", DeprecationWarning
        )
    if encrypted:
        raise NotImplementedError("Loading encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Load failed: src argument must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Load failed: src must be in [0, world_size)"

        # source party
        if comm.get().get_rank() == src:
            assert not (f is None and preloaded is None), "Load failed: f or preloaded should be specified"

            result = torch.load(f, **kwargs) if f else preloaded

            # Zero out the tensors / modules to hide loaded data from broadcast
            if torch.is_tensor(result):
                result_zeros = result.new_zeros(result.size())
            elif isinstance(result, torch.nn.Module):
                result_zeros = copy.deepcopy(result)
                result_zeros.set_all_parameters(0)
            else:
                result = comm.get().broadcast_obj(-1, src)
                raise TypeError("Unrecognized load type %s" % type(result))

            comm.get().broadcast_obj(result_zeros, src)

        # Non-source party
        else:
            result = comm.get().broadcast_obj(None, src)
            if isinstance(result, int) and result == -1:
                raise TypeError("Unrecognized load type from src party")

        if torch.is_tensor(result):
            result = crypten.cryptensor(result, src=src)
        # TODO: Encrypt modules before returning them
        # elif isinstance(result, torch.nn.Module):
        #     result = crypten.nn.from_pytorch(result, src=src)
        result.src = src
        return result


def load(
    f,
    preloaded=None,
    encrypted=False,
    dummy_model=None,
    src=0,
    load_closure=torch.load,
    **kwargs
):
    """
    Loads an object saved with `torch.save()` or `crypten.save_from_party()`.
    Note: this function is deprecated; please use load_from_party instead.
    """
    warnings.warn(
        "The current 'load' function is deprecated, and will be removed soon. "
        "To continue using current 'load' functionality, please use the "
        "'load_from_party' function instead.",
        DeprecationWarning,
    )
    load_from_party(f, preloaded, encrypted, dummy_model, src, load_closure, **kwargs)


def save_from_party(obj, f, src=0, save_closure=torch.save, **kwargs):
    """
    Saves a CrypTensor or PyTorch tensor to a file.

    Args:
        obj: The CrypTensor or PyTorch tensor to be saved
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        src: The source party that writes data to the specified file.
        save_closure: Custom save function that matches the interface of `torch.save`,
        to be used when the tensor is saved with a custom load function in
        `crypten.load_from_party`. Additional kwargs are passed on to the closure.
    """
    if is_encrypted_tensor(obj):
        raise NotImplementedError("Saving encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Save failed: src must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Save failed: src must be an integer in [0, world_size)"

        if comm.get().get_rank() == src:
            save_closure(obj, f, **kwargs)

    # Implement barrier to avoid race conditions that require file to exist
    comm.get().barrier()


def save(obj, f, src=0, save_closure=torch.save, **kwargs):
    """
    Saves a CrypTensor or PyTorch tensor to a file.
    Note: this function is deprecated, please use save_from_party instead
    """
    warnings.warn(
        "The current 'save' function is deprecated, and will be removed soon. "
        "To continue using current 'save' functionality, please use the "
        "'save_from_party' function instead.",
        DeprecationWarning,
    )
    save_from_party(obj, f, src, save_closure, **kwargs)


def where(condition, input, other):
    """
    Return a tensor of elements selected from either `input` or `other`, depending
    on `condition`.
    """
    if is_encrypted_tensor(condition):
        return condition * input + (1 - condition) * other
    elif torch.is_tensor(condition):
        condition = condition.float()
    return input * condition + other * (1 - condition)


def cat(tensors, dim=0):
    """
    Concatenates the specified CrypTen `tensors` along dimension `dim`.
    """
    assert isinstance(tensors, list), "input to cat must be a list"
    assert all(isinstance(t, CrypTensor) for t in tensors), "inputs must be CrypTensors"
    tensor_types = [get_cryptensor_type(t) for t in tensors]
    assert all(
        ttype == tensor_types[0] for ttype in tensor_types
    ), "cannot concatenate CrypTensors with different underlying types"
    if len(tensors) == 1:
        return tensors[0]
    return type(tensors[0]).cat(tensors, dim=dim)


def stack(tensors, dim=0):
    """
    Stacks the specified CrypTen `tensors` along dimension `dim`. In contrast to
    `crypten.cat`, this adds a dimension to the result tensor.
    """
    assert isinstance(tensors, list), "input to stack must be a list"
    assert all(isinstance(t, CrypTensor) for t in tensors), "inputs must be CrypTensors"
    tensor_types = [get_cryptensor_type(t) for t in tensors]
    assert all(
        ttype == tensor_types[0] for ttype in tensor_types
    ), "cannot stack CrypTensors with different underlying types"
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    return type(tensors[0]).stack(tensors, dim=dim)


def rand(*sizes, cryptensor_type=None):
    """
    Returns a tensor with elements uniformly sampled in [0, 1).
    """
    if cryptensor_type is None:
        cryptensor_type = get_default_cryptensor_type()
    return __CRYPTENSOR_TYPES__[cryptensor_type].rand(*sizes)


def bernoulli(tensor, cryptensor_type=None):
    """
    Returns a tensor with elements in {0, 1}. The i-th element of the
    output will be 1 with probability according to the i-th value of the
    input tensor.
    """
    return rand(tensor.size(), cryptensor_type=cryptensor_type) < tensor


# expose classes and functions in package:
__all__ = [
    "CrypTensor",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "debug",
    "init",
    "init_thread",
    "mpc",
    "nn",
    "uninit",
]
