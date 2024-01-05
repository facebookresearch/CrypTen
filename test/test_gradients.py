#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import itertools
import logging
import unittest
from collections import namedtuple

import crypten
import torch
import torch.nn.functional as F
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.gradients import AutogradContext
from test.multiprocess_test_case import (
    get_random_test_tensor,
    MultiProcessTestCase,
    onehot,
)


# Sizes for tensor operations
SIZES = [
    (),
    (1,),
    (3,),
    (1, 1),
    (1, 3),
    (3, 1),
    (3, 3),
    (1, 1, 1),
    (1, 1, 3),
    (1, 3, 1),
    (3, 1, 1),
    (3, 3, 3),
    (1, 1, 1, 1),
    (1, 1, 3, 1),
    (3, 3, 3, 3),
]


class TestGradients:
    """
    This class tests all autograd functions implemented in gradients.py.
    """

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communicator
        if self.rank >= 0:
            crypten.init()

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def _check_forward_backward(
        self,
        func_name,
        input_tensor,
        *args,
        torch_func_name=None,
        msg=None,
        addl_args=None,
        **kwargs,
    ):
        """Checks forward and backward against PyTorch

        Args:
            func_name (str): PyTorch/CrypTen function name
            input_tensor (torch.tensor): primary input
            args (list): contains arguments for function
            msg (str): additional message for mismatch
            kwargs (list): keyword arguments for function
        """

        if msg is None:
            msg = f"{func_name} grad_fn incorrect"

        input = input_tensor.clone()
        input.requires_grad = True
        input_encr = crypten.cryptensor(input, requires_grad=True)

        crypten_kwargs = copy.deepcopy(kwargs)
        if addl_args is not None:
            for item, val in addl_args.items():
                crypten_kwargs[item] = val

        for private in [False, True]:
            input.grad = None
            input_encr.grad = None
            args = self._set_grad_to_zero(args)
            args_encr = self._set_grad_to_zero(list(args), make_private=private)

            # obtain torch function
            if torch_func_name is not None:
                torch_func = self._get_torch_func(torch_func_name)
            else:
                torch_func = self._get_torch_func(func_name)

            reference = torch_func(input, *args, **kwargs)
            encrypted_out = getattr(input_encr, func_name)(*args_encr, **crypten_kwargs)

            # extract argmax output for max / min with keepdim=False
            if isinstance(encrypted_out, (list, tuple)):
                reference = reference[0]
                encrypted_out = encrypted_out[0]

            self._check(encrypted_out, reference, msg + " in forward")

            # check backward pass
            grad_output = get_random_test_tensor(
                max_value=2, size=reference.size(), is_float=True
            )
            grad_output_encr = crypten.cryptensor(grad_output)
            reference.backward(grad_output)
            encrypted_out.backward(grad_output_encr)
            self._check(input_encr.grad, input.grad, msg + " in backward")
            for i, arg_encr in enumerate(args_encr):
                if crypten.is_encrypted_tensor(arg_encr):
                    self._check(arg_encr.grad, args[i].grad, msg + " in backward args")

    def _set_grad_to_zero(self, args, make_private=False):
        """Sets gradients for args to zero

        Args:
            args (list of torch.tensors): contains arguments
            make_private (bool): encrypt args using CrypTensor
        """
        args_zero_grad = []

        for arg in args:
            if is_float_tensor(arg) and make_private:
                arg = crypten.cryptensor(arg, requires_grad=True)
            elif is_float_tensor(arg):
                arg.requires_grad = True
                arg.grad = None

            args_zero_grad.append(arg)

        return args_zero_grad

    def _get_torch_func(self, func_name):
        """Returns PyTorch function from tensor or functional API"""
        if hasattr(torch.Tensor, func_name):
            return getattr(torch.Tensor, func_name)
        elif hasattr(F, func_name):
            return getattr(F, func_name)
        else:
            raise ValueError("unknown PyTorch function: %s" % func_name)

    def test_arithmetic(self):
        """Tests arithmetic functions with broadcasting."""
        arithmetic_functions = ["add", "sub", "mul"]
        for func in arithmetic_functions:
            # Test on operator
            ofunc = "__" + func + "__"

            # Test both left functions and right functions
            rfunc = ofunc[:2] + "r" + ofunc[2:]

            # Test on both float inputs and tensor inputs
            for use_tensor in [False, True]:
                for size1 in SIZES:
                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    if use_tensor:
                        for size2 in SIZES:
                            tensor2 = get_random_test_tensor(size=size2, is_float=True)
                            self._check_forward_backward(func, tensor1, tensor2)
                            self._check_forward_backward(ofunc, tensor1, tensor2)
                            self._check_forward_backward(rfunc, tensor1, tensor2)
                    else:
                        scalar = 2.0
                        self._check_forward_backward(func, tensor1, scalar)
                        self._check_forward_backward(ofunc, tensor1, scalar)
                        self._check_forward_backward(rfunc, tensor1, scalar)

    def test_div(self):
        self._div_helper("div")

    def test_truediv(self):
        self._div_helper("__truediv__")

    def test_rtruediv(self):
        self._div_helper("__rtruediv__")

    def _div_helper(self, func):
        for size1 in SIZES:
            tensor1 = get_random_test_tensor(size=size1, is_float=True)
            for size2 in SIZES:
                tensor2 = get_random_test_tensor(
                    min_value=0.5, size=size2, is_float=True
                )  # do not divide by value very close to zero
                if func == "__rtruediv__":
                    # denominator is first argument for rtruediv
                    self._check_forward_backward(func, tensor2, tensor1)
                else:
                    self._check_forward_backward(func, tensor1, tensor2)

            if func == "__rtruediv__":
                self._check_forward_backward(func, torch.tensor(2.0), tensor2)
            else:
                self._check_forward_backward(func, tensor1, 2.0)

    def test_sum_mean_reductions(self):
        reductions = ["sum", "mean"]
        self._reductions_helper(reductions)

    def test_max_min_reductions_pairwise(self):
        reductions = ["max", "min"]
        self._reductions_helper(reductions, "pairwise")

    def test_max_min_reductions_log_reduction(self):
        reductions = ["max", "min"]
        self._reductions_helper(reductions, "log_reduction")

    def test_max_min_reductions_double_log_reduction(self):
        reductions = ["max", "min"]
        self._reductions_helper(reductions, "double_log_reduction")

    def test_max_min_reductions_accelerated_cascade(self):
        reductions = ["max", "min"]
        self._reductions_helper(reductions, "accelerated_cascade")

    def _reductions_helper(self, input_reductions, method=None):
        """Tests input reductions on tensors of various sizes."""
        for size in SIZES[: min(5, len(SIZES))]:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for reduction in input_reductions:
                if method is None:
                    self._check_forward_backward(reduction, tensor)
                else:
                    with cfg.temp_override({"functions.max_method": method}):
                        self._check_forward_backward(reduction, tensor)

                # Check dim 0 if tensor is 0-dimensional
                dims = 1 if tensor.dim() == 0 else tensor.dim()
                for dim in range(dims):

                    # check when keepdim is not provided as a kwarg
                    if method is None:
                        self._check_forward_backward(reduction, tensor, dim=dim)
                    else:
                        with cfg.temp_override({"functions.max_method": method}):
                            self._check_forward_backward(reduction, tensor, dim=dim)

                    # check when keepdim is provided as a kwarg
                    for keepdim in [False, True]:
                        if method is None:
                            self._check_forward_backward(
                                reduction, tensor, dim, keepdim=keepdim
                            )
                            self._check_forward_backward(
                                reduction, tensor, dim=dim, keepdim=keepdim
                            )
                        else:
                            with cfg.temp_override({"functions.max_method": method}):
                                self._check_forward_backward(
                                    reduction, tensor, dim, keepdim=keepdim
                                )
                                self._check_forward_backward(
                                    reduction, tensor, dim=dim, keepdim=keepdim
                                )

    def test_matmul(self):
        """Test matmul with broadcasting."""
        matmul_sizes = [(1, 1), (1, 5), (5, 1), (5, 5)]
        batch_dims = [(), (1,), (5,), (1, 1), (1, 5), (5, 5)]

        matched_sizes = [
            ((1,), (1,)),
            ((10,), (10,)),
            ((10,), (10, 5)),
            ((5, 10), (10,)),
        ]

        matmul_funcs = ["matmul", "__matmul__", "__imatmul__"]
        torch_funcs = ["matmul", "__matmul__", "__matmul__"]
        for i, func in enumerate(matmul_funcs):
            for size in matmul_sizes:
                for batch1, batch2 in itertools.combinations(batch_dims, 2):
                    size1 = (*batch1, *size)
                    size2 = (*batch2, *size)

                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=size2, is_float=True)
                    tensor2 = tensor2.transpose(-2, -1)
                    self._check_forward_backward(
                        func, tensor1, tensor2, torch_func_name=torch_funcs[i]
                    )

            for sizes in matched_sizes:
                tensor1 = get_random_test_tensor(size=sizes[0], is_float=True)
                tensor2 = get_random_test_tensor(size=sizes[1], is_float=True)

                self._check_forward_backward(
                    func, tensor1, tensor2, torch_func_name=torch_funcs[i]
                )

    def test_unary_functions(self):
        """Test unary functions on tensors of various sizes."""
        unary_functions = [
            "neg",
            "__neg__",
            "exp",
            "reciprocal",
            "abs",
            "__abs__",
            "sign",
            "relu",
            "sin",
            "cos",
            "sigmoid",
            "tanh",
            "log",
            "sqrt",
        ]
        pos_only_functions = ["log", "sqrt"]
        for func in unary_functions:
            for size in SIZES:
                tensor = get_random_test_tensor(size=size, is_float=True)

                # Make tensor positive when positive inputs are required
                if func in pos_only_functions:
                    tensor = tensor.abs()

                self._check_forward_backward(func, tensor)

    def test_hardtanh(self):
        tensor = torch.arange(-10, 10, dtype=torch.float32)
        for minval in range(-10, 10):
            for maxval in range(minval, 11):
                self._check_forward_backward("hardtanh", tensor, minval, maxval)
        self._check_forward_backward("relu6", tensor)

    def test_inplace_warning(self):
        """Tests that a warning is thrown that indicates that the `inplace` kwarg
        is ignored when a function is called with `inplace=True`
        """
        tensor = get_random_test_tensor(is_float=True)
        encrypted = crypten.cryptensor(tensor)

        functions = ["dropout", "_feature_dropout"]
        for func in functions:
            warning_str = (
                f"CrypTen {func} does not support inplace computation during training."
            )
            with self.assertLogs(logger=logging.getLogger(), level="WARNING") as cm:
                getattr(encrypted, func)(inplace=True)
            self.assertTrue(f"WARNING:root:{warning_str}" in cm.output)

    def test_dot_ger(self):
        """Test inner and outer products of encrypted tensors."""
        for length in range(1, 10):
            tensor1 = get_random_test_tensor(size=(length,), is_float=True)
            tensor2 = get_random_test_tensor(size=(length,), is_float=True)

            self._check_forward_backward("dot", tensor1, tensor2)
            self._check_forward_backward("ger", tensor1, tensor2)

    def test_squeeze_unsqueeze(self):
        """Test addition and removal of tensor dimensions"""
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)

            self._check_forward_backward("squeeze", tensor)
            for dim in range(tensor.dim()):
                self._check_forward_backward("squeeze", tensor, dim)
                self._check_forward_backward("unsqueeze", tensor, dim)

            # Check unsqueeze on last dimension
            self._check_forward_backward("unsqueeze", tensor, tensor.dim())

    def test_softmax(self):
        """Test softmax"""
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)

            # Check dim 0 if tensor is 0-dimensional
            dims = 1 if tensor.dim() == 0 else tensor.dim()
            for dim in range(dims):
                self._check_forward_backward("softmax", tensor, dim)

    def test_log_softmax(self):
        """Test log_softmax"""
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)

            # Check dim 0 if tensor is 0-dimensional
            dims = 1 if tensor.dim() == 0 else tensor.dim()
            for dim in range(dims):
                self._check_forward_backward("log_softmax", tensor, dim)

    def test_transpose(self):
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)

            if tensor.dim() == 2:  # t() asserts dim == 2
                self._check_forward_backward("t", tensor)

            for dim0 in range(tensor.dim()):
                for dim1 in range(tensor.dim()):
                    self._check_forward_backward("transpose", tensor, dim0, dim1)

    def test_permute(self):
        for ndims in range(5):
            size = tuple([3] * ndims)
            tensor = get_random_test_tensor(size=size, is_float=True)

            for perm in itertools.permutations(list(range(ndims))):
                self._check_forward_backward("permute", tensor, perm)

    def test_conv1d_smaller_signal_one_channel(self):
        self._conv1d(5, 1)

    def test_conv1d_smaller_signal_many_channels(self):
        self._conv1d(5, 5)

    def test_conv1d_larger_signal_one_channel(self):
        self._conv1d(16, 1)

    def test_conv1d_larger_signal_many_channels(self):
        self._conv1d(16, 5)

    def _conv1d(self, signal_size, in_channels):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        nout_channels = [1, 5]
        kernel_sizes = [1, 2, 3]
        paddings = [0, 1]
        strides = [1, 2]
        dilations = [1, 2]
        groupings = [1, 2]

        for (
            batches,
            kernel_size,
            out_channels,
            padding,
            stride,
            dilation,
            groups,
        ) in itertools.product(
            nbatches,
            kernel_sizes,
            nout_channels,
            paddings,
            strides,
            dilations,
            groupings,
        ):
            # TODO: Fix conv1d gradient in this case:
            if in_channels > 1 and groups > 1:
                continue

            size = (batches, in_channels * groups, signal_size)
            signal = get_random_test_tensor(size=size, is_float=True)

            kernel_size = (out_channels * groups, in_channels, kernel_size)
            kernel = get_random_test_tensor(size=kernel_size, is_float=True)

            self._check_forward_backward(
                "conv1d",
                signal,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

    def test_conv2d_square_image_one_channel(self):
        self._conv2d((5, 5), 1)

    def test_conv2d_square_image_many_channels(self):
        self._conv2d((5, 5), 5)

    def test_conv2d_rectangular_image_one_channel(self):
        self._conv2d((16, 7), 1)

    def test_conv2d_rectangular_image_many_channels(self):
        self._conv2d((16, 7), 5)

    def _conv2d(self, image_size, in_channels):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        ochannels = [1, 3]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]
        dilations = [1, 2, (1, 2)]
        groupings = [1, 2]

        for (
            batches,
            kernel_size,
            out_channels,
            padding,
            stride,
            dilation,
            groups,
        ) in itertools.product(
            nbatches, kernel_sizes, ochannels, paddings, strides, dilations, groupings
        ):
            # TODO: Fix conv2d gradient in this case:
            if in_channels > 1 and groups > 1:
                continue

            size = (batches, in_channels * groups, *image_size)
            image = get_random_test_tensor(size=size, is_float=True)

            kernel_size = (out_channels * groups, in_channels, *kernel_size)
            kernel = get_random_test_tensor(size=kernel_size, is_float=True)

            self._check_forward_backward(
                "conv2d",
                image,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

    def test_max_pool2d(self):
        """Tests max pooling gradient"""
        self._check_pooling("max_pool2d")

    def test_avg_pool2d(self):
        """Tests average pooling gradient"""
        self._check_pooling("avg_pool2d")

    def _check_pooling(self, func):
        """Helper for testing pooling gradients to avoid test timeouts"""
        image_sizes = [(5, 5), (6, 7)]
        nchannels = [1, 3]
        nbatches = [1, 3]

        kernel_sizes = [1, 2, (2, 3)]
        paddings = [1, (0, 0)]
        strides = [1, (2, 2)]
        dilations = [1, 2]

        ceil_modes = [False, True] if func == "max_pool2d" else [False]

        for image_size, channels, batches, kernel_size in itertools.product(
            image_sizes, nchannels, nbatches, kernel_sizes
        ):
            size = (batches, channels, *image_size)
            image = get_random_test_tensor(size=size, is_float=True)

            for padding, stride, ceil_mode in itertools.product(
                paddings, strides, ceil_modes
            ):
                # Skip invalid padding sizes
                if kernel_size == 1 and padding == 1:
                    continue
                if func == "max_pool2d":
                    for dilation in dilations:
                        self._check_max_pool2d_forward_backward(
                            image, kernel_size, padding, stride, dilation, ceil_mode
                        )
                else:
                    self._check_forward_backward(
                        func, image, kernel_size, padding=padding, stride=stride
                    )

    def _check_max_pool2d_forward_backward(
        self, image, kernel_size, padding, stride, dilation, ceil_mode, tol=0.1
    ):
        """Checks forward and backward are for max pool 2d.
        Verifies gradients by checking sum of non-matching elements to account for
        differences in tie resolution in max between PyTorch and CrypTen:
        PyTorch returns smallest index for max entries,
        whereas CrypTen returns a random index.

        Args:
            image (torch.tensor): input
            kernel_size (tuple of ints): size of the window over which to compute max
            padding (int or tuple of ints): implicit zero padding to added on both sides
            stride (int or tuple of ints): the stride of the window
            ceil_mode (bool): determines whether output size is rounded down or up
        """
        # check forward
        image = image.clone()
        image.requires_grad = True
        image_enc = crypten.cryptensor(image, requires_grad=True)

        out = torch.nn.functional.max_pool2d(
            image,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        out_enc = image_enc.max_pool2d(
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        if out.isinf().any():
            # PyTorch can produce improperly sized outputs with Inf values using ceil_mode in some cases
            if ceil_mode:
                return
            self.assertTrue(
                out.size() == out_enc.size(), "max_pool2d forward incorrect"
            )
            return  # backward will break if output is -inf
        else:
            self._check(out_enc, out, "max_pool2d forward incorrect")

        # check backward
        grad_output = get_random_test_tensor(size=out.size(), is_float=True)
        grad_output_enc = crypten.cryptensor(grad_output)
        out.backward(grad_output)
        out_enc.backward(grad_output_enc)

        # check sum of non-matching gradient entries
        crypten_grad = image_enc.grad.get_plain_text()
        non_matching_indices = (image.grad - crypten_grad).abs() > tol
        sum_is_close = (
            crypten_grad[non_matching_indices].sum()
            - image.grad[non_matching_indices].sum()
        ) < tol
        if not sum_is_close:
            msg = "max_pool2d backward failed"
            logging.info(msg)
            logging.info(f"Result: crypten image gradient {crypten_grad}")
            logging.info(f"Result - Reference {image.grad - crypten_grad}")
            self.assertTrue(sum_is_close, msg=msg)

    def test_square(self) -> None:
        """Tests square function gradient.
        Note: torch pow(2) is used to verify gradient,
            since PyTorch does not implement square().
        """
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)
            tensor.requires_grad = True
            tensor_encr = crypten.cryptensor(tensor, requires_grad=True)

            out = tensor.pow(2)
            out_encr = tensor_encr.square()
            self._check(out_encr, out, f"square forward failed with size {size}")

            grad_output = get_random_test_tensor(size=out.shape, is_float=True)
            out.backward(grad_output)
            out_encr.backward(crypten.cryptensor(grad_output))
            self._check(
                tensor_encr.grad,
                tensor.grad,
                f"square backward failed with size {size}",
            )

    def test_pow(self) -> None:
        self._pow_helper("pow")

    def test_magic_pow(self) -> None:
        self._pow_helper("__pow__")

    def _pow_helper(self, pow_fn):
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, min_value=0.5, is_float=True)
            for power in [-3, -2, -1, 0, 1, 2, 3]:
                self._check_forward_backward(pow_fn, tensor, power)
                self._check_forward_backward(pow_fn, tensor, float(power))

    def test_norm(self) -> None:
        """Tests p-norm"""
        self.default_tolerance *= 2  # Increase tolerance for norm test
        for p in [1, 1.5, 2, 3, float("inf"), "fro"]:
            tensor = get_random_test_tensor(max_value=2, size=(3, 3, 3), is_float=True)

            self._check_forward_backward("norm", tensor, p=p)
            for dim in [0, 1, 2]:
                self._check_forward_backward("norm", tensor, p=p, dim=dim)

    def test_pad(self) -> None:
        """Tests padding"""
        sizes = [(1,), (5,), (1, 1), (5, 5), (5, 5, 5), (5, 3, 32, 32)]
        pads = [
            # (0, 0, 0, 0), NOTE: Pytorch backward fails when padding is all 0s
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 1, 1),
            (2, 2, 1, 1),
            (2, 2, 2, 2),
        ]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for pad in pads:
                if tensor.dim() < 2:
                    pad = pad[:2]

                    # NOTE: Pytorch backward fails when padding is all 0s
                    if pad[0] == 0 and pad[1] == 0:
                        continue

                for value in [0, 1, 10]:
                    self._check_forward_backward("pad", tensor, pad, value=value)

    def test_clone(self) -> None:
        """Tests shallow_copy and clone of encrypted tensors."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("clone", tensor)

    def test_cat_stack(self) -> None:
        for module in [crypten, torch]:  # torch.cat on CrypTensor runs crypten.cat
            for func in ["cat", "stack"]:
                for dimensions in range(1, 5):
                    size = [5] * dimensions
                    for num_tensors in range(1, 5):
                        for dim in range(dimensions):
                            tensors = [
                                get_random_test_tensor(size=size, is_float=True)
                                for _ in range(num_tensors)
                            ]
                            encrypted_tensors = [
                                crypten.cryptensor(t, requires_grad=True)
                                for t in tensors
                            ]
                            for i in range(len(tensors)):
                                tensors[i].grad = None
                                tensors[i].requires_grad = True
                                encrypted_tensors[i].grad = None
                                encrypted_tensors[i].requires_grad = True

                            # Forward
                            reference = getattr(torch, func)(tensors, dim=dim)
                            encrypted_out = getattr(module, func)(
                                encrypted_tensors, dim=dim
                            )
                            self._check(
                                encrypted_out, reference, f"{func} forward failed"
                            )

                            # Backward
                            grad_output = get_random_test_tensor(
                                size=reference.size(), is_float=True
                            )
                            encrypted_grad_output = crypten.cryptensor(grad_output)

                            reference.backward(grad_output)
                            encrypted_out.backward(encrypted_grad_output)
                            for i in range(len(tensors)):
                                self._check(
                                    encrypted_tensors[i].grad,
                                    tensors[i].grad,
                                    f"{func} backward failed",
                                )

    def test_dropout(self) -> None:
        """Tests forward for dropout"""
        # Create a separate test for dropout since it cannot use the
        # regular forward function
        # There's no need to check backwards since PyTorch backwards fails
        all_prob_values = [x * 0.2 for x in range(0, 5)]
        for dropout_fn in ["dropout", "_feature_dropout"]:
            for prob in all_prob_values:
                for size in [(5, 10), (5, 10, 15), (5, 10, 15, 20)]:
                    for use_zeros in [False, True]:
                        tensor = get_random_test_tensor(
                            size=size, ex_zero=True, min_value=1.0, is_float=True
                        )
                        if use_zeros:
                            # turn the first row to all zeros
                            index = [1] + [
                                slice(0, tensor.size(i)) for i in range(1, tensor.dim())
                            ]
                            tensor[index] = 0.0

                        encr_tensor = crypten.cryptensor(tensor, requires_grad=True)
                        encr_tensor_out = getattr(encr_tensor, dropout_fn)(p=prob)
                        dropout_tensor = encr_tensor_out.get_plain_text()

                        # Check the scaling for non-zero elements
                        scaled_tensor = tensor / (1 - prob)
                        reference = dropout_tensor.where(
                            dropout_tensor == 0.0, scaled_tensor
                        )
                        self._check(
                            encr_tensor_out,
                            reference,
                            "dropout failed with size {}, use_zeros {}, and "
                            "probability {}".format(size, use_zeros, prob),
                        )

    def test_batchnorm(self) -> None:
        """
        Tests batchnorm forward and backward steps with training on / off.
        """
        tolerance = 0.1
        sizes = [(8, 5), (16, 3), (32, 5), (8, 6, 4), (8, 4, 3, 5)]
        torch.autograd.set_detect_anomaly(True)
        for size in sizes:
            for is_training in (False, True):

                # sample input data, weight, and bias:
                tensor = get_random_test_tensor(size=size, is_float=True)
                encrypted_input = crypten.cryptensor(tensor)
                C = size[1]
                weight = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                bias = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                weight.requires_grad = True
                bias.requires_grad = True

                # dimensions over which means and variances are computed:
                stats_dimensions = list(range(tensor.dim()))
                stats_dimensions.pop(1)

                # dummy running mean and variance:
                running_mean = tensor.mean(stats_dimensions).detach()
                running_var = tensor.var(stats_dimensions).detach()
                enc_running_mean = crypten.cryptensor(running_mean)
                enc_running_var = crypten.cryptensor(running_var)

                # compute reference output:
                tensor.requires_grad = True
                reference = torch.nn.functional.batch_norm(
                    tensor,
                    running_mean,
                    running_var,
                    weight=weight,
                    bias=bias,
                    training=is_training,
                )

                # compute CrypTen output:
                encrypted_input.requires_grad = True
                ctx = AutogradContext()
                batch_norm_fn = crypten.gradients.get_grad_fn("batchnorm")
                with crypten.no_grad():
                    encrypted_out = batch_norm_fn.forward(
                        ctx,
                        encrypted_input,
                        weight,
                        bias,
                        training=is_training,
                        running_mean=enc_running_mean,
                        running_var=enc_running_var,
                    )

                # check forward
                self._check(
                    encrypted_out,
                    reference,
                    "batchnorm forward failed with training "
                    f"{is_training} on {tensor.dim()}-D",
                    tolerance=tolerance,
                )

                # check backward (input, weight, and bias gradients):
                grad_input = get_random_test_tensor(
                    size=reference.size(), is_float=True
                )
                reference.backward(grad_input)
                with crypten.no_grad():
                    enc_grad_input = crypten.cryptensor(grad_input)
                    encrypted_grad = batch_norm_fn.backward(ctx, enc_grad_input)
                TorchGrad = namedtuple("TorchGrad", ["name", "value"])
                torch_gradients = [
                    TorchGrad("input gradient", tensor.grad),
                    TorchGrad("weight gradient", weight.grad),
                    TorchGrad("bias gradient", bias.grad),
                ]
                for i, torch_gradient in enumerate(torch_gradients):
                    self._check(
                        encrypted_grad[i],
                        torch_gradient.value,
                        f"batchnorm backward {torch_gradient.name} failed "
                        f"with training {is_training} on {tensor.dim()}-D",
                        tolerance=tolerance,
                    )

    def test_cross_entropy(self) -> None:
        """Tests cross_entropy and binary_cross_entropy"""
        sizes = [(3, 2), (8, 4), (5, 10)]
        losses = [
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
            "cross_entropy",
        ]

        for size, loss in itertools.product(sizes, losses):
            for skip_forward in [False, True]:
                batch_size, num_targets = size
                if loss in ["binary_cross_entropy", "binary_cross_entropy_with_logits"]:
                    if loss == "binary_cross_entropy":
                        tensor = get_random_test_tensor(
                            size=(batch_size,), max_value=0.998, is_float=True
                        )
                        tensor = tensor.abs().add_(0.001)
                    else:
                        tensor = get_random_test_tensor(
                            size=(batch_size,), is_float=True
                        )

                    target = get_random_test_tensor(size=(batch_size,), is_float=True)
                    target = target.gt(0.0).float()
                    target_encr = crypten.cryptensor(target)
                else:
                    tensor = get_random_test_tensor(size=size, is_float=True)
                    target = get_random_test_tensor(
                        size=(batch_size,), max_value=num_targets - 1
                    )
                    target = onehot(target.abs(), num_targets=num_targets)
                    target_encr = crypten.cryptensor(target)
                    # CrypTen, unlike PyTorch, uses one-hot targets
                    target = target.argmax(1)

                # forward
                tensor.requires_grad = True
                tensor_encr = crypten.cryptensor(tensor, requires_grad=True)
                reference = getattr(torch.nn.functional, loss)(tensor, target)
                out_encr = getattr(tensor_encr, loss)(
                    target_encr, skip_forward=skip_forward
                )
                if not skip_forward:
                    self._check(out_encr, reference, f"{loss} forward failed")

                # backward
                reference.backward()
                out_encr.backward()
                self._check(tensor_encr.grad, tensor.grad, f"{loss} backward failed")

    def test_rappor_loss(self) -> None:
        """Tests RAPPOR Loss"""
        sizes = [(3,), (8,), (5,)]
        alphas = [0.1, 0.3, 0.4]

        for size, alpha in itertools.product(sizes, alphas):
            for skip_forward in [True, False]:
                tensor = get_random_test_tensor(size=size, is_float=True)

                target = get_random_test_tensor(size=size, is_float=True)
                target = target.gt(0.0).float()
                target_encr = crypten.cryptensor(target)

                # forward
                tensor.requires_grad = True
                tensor_encr = crypten.cryptensor(tensor, requires_grad=True)

                reference = tensor.sigmoid()
                reference = alpha * reference + (1 - alpha) * (1 - reference)

                reference = torch.nn.functional.binary_cross_entropy(reference, target)
                out_encr = tensor_encr.rappor_loss(
                    target_encr, alpha, skip_forward=skip_forward
                )

                if not skip_forward:
                    self._check(out_encr, reference, "rappor_loss forward failed")

                # backward
                reference.backward()
                out_encr.backward()
                self._check(
                    tensor_encr.grad, tensor.grad, "rappor_loss backward failed"
                )

    def test_cosine_similarity(self) -> None:
        """Tests cosine_similarity"""
        for size in SIZES:
            tensor0 = get_random_test_tensor(size=size, is_float=True)
            tensor1 = get_random_test_tensor(size=size, is_float=True)

            # Check dim 0 if tensor is 0-dimensional
            dims = 1 if len(size) == 0 else len(size)
            for dim in range(dims):
                self._check_forward_backward(
                    "cosine_similarity", tensor0, tensor1, dim=dim
                )

    def test_view_reshape(self) -> None:
        """Tests view and reshape gradients"""
        size_to_views = {
            (10,): [(5, 2), (1, 10)],
            (10, 5): [(50), (2, 5, 5)],
            (5, 10, 8): [(400), (50, 8), (5, 5, 2, 8)],
        }

        for size in size_to_views:
            for view in size_to_views[size]:
                tensor = get_random_test_tensor(size=size, is_float=True)
                self._check_forward_backward("view", tensor, view)
                self._check_forward_backward("reshape", tensor, view)

    def test_narrow_flatten(self) -> None:
        """Tests narrow and flatten gradients"""
        sizes = [(10,), (5, 4), (10, 6, 8)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("flatten", tensor)
            for dim in range(tensor.dim()):
                self._check_forward_backward("narrow", tensor, dim, 0, 2)
                self._check_forward_backward("narrow", tensor, dim, 1, 3)

    def test_flip(self) -> None:
        """Tests flip gradient"""
        sizes = [(2, 3, 7, 2), (5, 10, 15)]
        flips = [(0, 2, 1), (0, 1)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for flip in flips:
                self._check_forward_backward("flip", tensor, flip)

    def test_gather_scatter(self) -> None:
        """Tests gather and scatter gradients"""
        sizes = [(2, 2), (3, 5), (3, 5, 10)]
        indices = [[0, 1, 0, 0], [0, 1, 0, 0, 1] * 3, [0, 0, 1] * 50]
        dims = [0, 1]
        funcs = ["scatter", "gather"]

        for dim, func in itertools.product(dims, funcs):
            for size, index in zip(sizes, indices):
                tensor = get_random_test_tensor(size=size, is_float=True)
                index = torch.tensor(index).reshape(tensor.shape)

                tensor.requires_grad = True
                tensor_encr = crypten.cryptensor(tensor, requires_grad=True)

                if func == "gather":
                    reference = getattr(tensor, func)(dim, index)
                    out_encr = getattr(tensor_encr, func)(dim, index)
                else:
                    src = get_random_test_tensor(size=index.shape, is_float=True)
                    reference = getattr(tensor, func)(dim, index, src)
                    out_encr = getattr(tensor_encr, func)(dim, index, src)

                self._check(
                    out_encr, reference, f"{func} forward failed with index {index}"
                )

                grad_out = get_random_test_tensor(size=reference.shape, is_float=True)
                grad_out_encr = crypten.cryptensor(grad_out)
                reference.backward(grad_out)
                out_encr.backward(grad_out_encr)

                self._check(
                    tensor_encr.grad,
                    tensor.grad,
                    f"{func} backward failed with index {index}",
                )

    def test_index_select(self) -> None:
        """Tests index_select gradients"""
        sizes = [(2, 2), (3, 5), (3, 5, 10), (4, 8, 2, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for dim in range(len(size)):
                for index_size in range(size[dim]):
                    index = get_random_test_tensor(
                        max_value=(size[dim] - 1),
                        min_value=0,
                        size=(index_size,),
                        is_float=False,
                    )
                    self._check_forward_backward("index_select", tensor, dim, index)

    def test_take(self) -> None:
        """Tests take gradients"""
        sizes = [(10,), (5, 10), (2, 5, 10)]
        indices = [[0], [0, 5], [0, 2, 5, 8]]

        for size, index in itertools.product(sizes, indices):
            tensor = get_random_test_tensor(size=size, is_float=True)
            index = torch.tensor(index)
            self._check_forward_backward("take", tensor, index)

    def test_roll(self) -> None:
        """Tests roll gradients"""
        sizes = [(1, 10), (5, 10), (2, 5, 10)]
        shifts = [1, 3, (1, 2)]
        dims = [0, 1, (0, 1)]

        for size, shift_dim in itertools.product(sizes, zip(shifts, dims)):
            shift, dim = shift_dim
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("roll", tensor, shift, dim)

    def test_cumsum(self) -> None:
        """Tests cumsum gradient"""
        sizes = [(), (10,), (5, 10), (2, 5, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for dim in range(tensor.dim()):
                self._check_forward_backward("cumsum", tensor, dim)

    def test_trace(self) -> None:
        """Tests trace gradient"""
        sizes = [(1, 1), (3, 3), (10, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("trace", tensor)

    def test_var(self) -> None:
        """Tests var gradient"""
        sizes = [(10,), (1, 10), (5, 10), (2, 5, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("var", tensor)
            for unbiased in [False, True]:
                self._check_forward_backward("var", tensor, unbiased=unbiased)
                for dim, keepdim in itertools.product(range(len(size)), [False, True]):
                    # skip dimensions with 1 element
                    if size[dim] == 1:
                        continue
                    self._check_forward_backward(
                        "var", tensor, dim, unbiased=unbiased, keepdim=keepdim
                    )

    def test_getitem(self) -> None:
        """Tests getitem gradient"""
        sizes = [(10,), (10, 1), (5, 10), (5, 2, 10)]
        indices = [0, 1, 3]

        for size, index in itertools.product(sizes, indices):
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("__getitem__", tensor, index)

    def test_pos_pow(self) -> None:
        """Test gradient crypten pos_pow"""
        for power in [3, -2, 1.75]:
            # ensure base is positive for pos_pow
            tensor = get_random_test_tensor(is_float=True, max_value=2) + 4
            tensor.requires_grad = True
            tensor_encr = crypten.cryptensor(tensor, requires_grad=True)

            reference = tensor.pow(power)
            out_encr = tensor_encr.pos_pow(power)
            self._check(
                out_encr, reference, f"pos_pow forward failed with power {power}"
            )

            grad_out = get_random_test_tensor(is_float=True)
            grad_out_encr = crypten.cryptensor(grad_out)
            reference.backward(grad_out)
            out_encr.backward(grad_out_encr)

            self._check(
                tensor_encr.grad,
                tensor.grad,
                f"pos_pow backward failed with power {power}",
            )

    def test_polynomial(self) -> None:
        for terms in range(1, 5):
            for encrypt_coeffs in [False, True]:
                tensor = get_random_test_tensor(is_float=True)
                tensor.requires_grad = True
                tensor_encr = crypten.cryptensor(tensor, requires_grad=True)

                coeffs_size = (terms,)
                coeffs = get_random_test_tensor(size=coeffs_size, is_float=True)

                reference = (
                    tensor.unsqueeze(0)
                    .pow(torch.arange(terms).add(1).view([terms] + [1] * terms))
                    .mul(coeffs.view([terms] + [1] * terms))
                    .sum(0)
                    .view(tensor.size())
                )
                if encrypt_coeffs:
                    coeffs = crypten.cryptensor(coeffs)
                out_encr = tensor_encr.polynomial(coeffs)
                self._check(out_encr, reference, "polynomial forward failed")

                grad_out = get_random_test_tensor(size=reference.size(), is_float=True)
                grad_out_encr = crypten.cryptensor(grad_out)
                reference.backward(grad_out)
                out_encr.backward(grad_out_encr)
                self._check(
                    tensor_encr.grad,
                    tensor.grad,
                    "polynomial backward failed",
                )


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestGradients):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestGradients):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTTP, self).tearDown()


class TestPTT(unittest.TestCase, TestGradients):
    def setUp(self) -> None:
        self.default_tolerance = 0.5
        self._original_backend = crypten.get_default_cryptensor_type()
        crypten.set_default_cryptensor_type("ptt")
        super(TestPTT, self).setUp()
        crypten.init()

    def tearDown(self) -> None:
        crypten.set_default_cryptensor_type(self._original_backend)
        super(TestPTT, self).setUp()


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
