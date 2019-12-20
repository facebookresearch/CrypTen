#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import unittest
from collections import namedtuple
from test.multiprocess_test_case import (
    MultiProcessTestCase,
    get_random_test_tensor,
    onehot,
)

import crypten
import torch
import torch.nn.functional as F
from crypten.autograd_cryptensor import AutogradContext, AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor


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


class TestGradients(object):
    """
        This class tests all functions of AutogradCrypTensor.
    """

    benchmarks_enabled = False

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
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

    def _check_forward_backward(
        self, func_name, input_tensor, *args, torch_func_name=None, msg=None, **kwargs
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
        input_encr = AutogradCrypTensor(crypten.cryptensor(input), requires_grad=True)

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
            encrypted_out = getattr(input_encr, func_name)(*args_encr, **kwargs)

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
            make_private (bool): encrypt args using AutogradCrypTensor
        """
        args_zero_grad = []

        for arg in args:
            if is_float_tensor(arg) and make_private:
                arg = AutogradCrypTensor(crypten.cryptensor(arg), requires_grad=True)
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
        funcs = ["div", "__truediv__", "__rtruediv__"]
        for func in funcs:
            for size1 in SIZES:
                tensor1 = get_random_test_tensor(size=size1, is_float=True)
                for size2 in SIZES:
                    tensor2 = get_random_test_tensor(
                        min_value=0.5, size=size2, is_float=True
                    )  # do not divide by value very close to zero
                    self._check_forward_backward(func, tensor1, tensor2)
                self._check_forward_backward(func, tensor1, 2.0)

    def test_reductions(self):
        """Tests reductions on tensors of various sizes."""
        reductions = ["sum", "mean", "max", "min"]
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for reduction in reductions:
                self._check_forward_backward(reduction, tensor)

                # Check dim 0 if tensor is 0-dimensional
                dims = 1 if tensor.dim() == 0 else tensor.dim()
                for dim in range(dims):
                    for keepdim in [False, True]:
                        self._check_forward_backward(
                            reduction, tensor, dim=dim, keepdim=keepdim
                        )

    def test_matmul(self):
        """Test matmul with broadcasting."""
        matmul_sizes = [(1, 1), (1, 5), (5, 1), (5, 5)]
        batch_dims = [(), (1,), (5,), (1, 1), (1, 5), (5, 5)]

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

    def test_conv2d(self):
        """Test convolution of encrypted tensor with public/private tensors."""
        image_sizes = [(5, 5), (16, 7)]
        nchannels = [1, 5]
        nbatches = [1, 3, 5]

        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]
        for image_size, in_channels, batches in itertools.product(
            image_sizes, nchannels, nbatches
        ):

            # sample input:
            size = (batches, in_channels, *image_size)
            image = get_random_test_tensor(size=size, is_float=True)

            for kernel_size, out_channels in itertools.product(kernel_sizes, nchannels):

                # Sample kernel
                kernel_size = (out_channels, in_channels, *kernel_size)
                kernel = get_random_test_tensor(size=kernel_size, is_float=True)

                for padding in paddings:
                    for stride in strides:
                        self._check_forward_backward(
                            "conv2d", image, kernel, stride=stride, padding=padding
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
        nchannels = [1, 3, 5]
        nbatches = [1, 5]

        kernel_sizes = [1, 2, (2, 3)]
        paddings = [1, (0, 0)]
        strides = [1, (2, 2)]

        for image_size, channels, batches, kernel_size in itertools.product(
            image_sizes, nchannels, nbatches, kernel_sizes
        ):
            size = (batches, channels, *image_size)
            image = get_random_test_tensor(size=size, is_float=True)

            for padding, stride in itertools.product(paddings, strides):
                # Skip invalid padding sizes
                if kernel_size == 1 and padding == 1:
                    continue
                self._check_forward_backward(
                    func, image, kernel_size, padding=padding, stride=stride
                )

    def test_square(self):
        """Tests square function gradient.
        Note: torch pow(2) is used to verify gradient,
            since PyTorch does not implement square().
        """
        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)
            tensor.requires_grad = True
            tensor_encr = AutogradCrypTensor(
                crypten.cryptensor(tensor), requires_grad=True
            )

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

    def test_pow(self):
        """Tests pow function"""
        for pow_fn in ["pow", "__pow__"]:
            for size in SIZES:
                tensor = get_random_test_tensor(
                    size=size, min_value=0.5, is_float=True
                )  # prevent division by values close to zero
                for power in [-3, -2, -1, 0, 1, 2, 3]:
                    self._check_forward_backward(pow_fn, tensor, power)
                    self._check_forward_backward(pow_fn, tensor, float(power))

    def test_norm(self):
        """Tests p-norm"""
        self.default_tolerance *= 2  # Increase tolerance for norm test
        for p in [1, 1.5, 2, 3, float("inf"), "fro"]:
            tensor = get_random_test_tensor(max_value=2, size=(3, 3, 3), is_float=True)

            self._check_forward_backward("norm", tensor, p=p)
            for dim in [0, 1, 2]:
                self._check_forward_backward("norm", tensor, p=p, dim=dim)

    def test_pad(self):
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

    def test_clone(self):
        """Tests shallow_copy and clone of encrypted tensors."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("clone", tensor)

    def test_cat_stack(self):
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
                            AutogradCrypTensor(crypten.cryptensor(t)) for t in tensors
                        ]
                        for i in range(len(tensors)):
                            tensors[i].grad = None
                            tensors[i].requires_grad = True
                            encrypted_tensors[i].grad = None
                            encrypted_tensors[i].requires_grad = True

                        # Forward
                        reference = getattr(torch, func)(tensors, dim=dim)
                        encrypted_out = getattr(crypten, func)(
                            encrypted_tensors, dim=dim
                        )
                        self._check(encrypted_out, reference, f"{func} forward failed")

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

    def test_dropout(self):
        """Tests forward and backward passes for dropout"""
        # Create a separate test for dropout since it cannot use the
        # regular forward function
        all_prob_values = [x * 0.2 for x in range(0, 5)]
        for dropout_fn in ["dropout", "_feature_dropout", "dropout2d", "dropout3d"]:
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

                        encr_tensor = AutogradCrypTensor(
                            crypten.cryptensor(tensor), requires_grad=True
                        )
                        encr_tensor_out = getattr(encr_tensor, dropout_fn)(p=prob)
                        dropout_tensor = encr_tensor_out.get_plain_text()
                        # Check the scaling for non-zero elements
                        scaled_tensor = tensor / (1 - prob)
                        reference = dropout_tensor.where(
                            dropout_tensor == 0, scaled_tensor
                        )
                        self._check(
                            encr_tensor_out,
                            reference,
                            "dropout failed with size {}, use_zeros {}, and "
                            "probability {}".format(size, use_zeros, prob),
                        )

                    # Check backward pass
                    grad_output = get_random_test_tensor(
                        max_value=2, size=reference.size(), is_float=True
                    )
                    grad_output_encr = crypten.cryptensor(grad_output)

                    # Do not check backward if pytorch backward fails
                    try:
                        reference.backward(grad_output)
                    except RuntimeError:
                        logging.info("skipped")
                        continue
                    encr_tensor_out.backward(grad_output_encr)

                    self._check(
                        encr_tensor.grad,
                        input.grad,
                        "dropout failed in backward with size {}, use_zeros {} and "
                        "probability {}".format(size, use_zeros, prob),
                    )

    def test_batchnorm(self):
        """
        Tests batchnorm forward and backward steps with training on / off.
        """
        # sizes for 1D, 2D, and 3D batchnorm
        # batch_size (dim=0) > 500 and increase tolerance to avoid flaky precision
        # errors in inv_var, which involves sqrt and reciprocal
        sizes = [(800, 5), (500, 8, 15), (600, 10, 3, 15)]
        tolerance = 0.5

        for size in sizes:
            for is_trainning in (False, True):
                tensor = get_random_test_tensor(size=size, is_float=True)
                tensor.requires_grad = True
                encrypted_input = crypten.cryptensor(tensor)

                C = size[1]
                weight = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                bias = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                weight.requires_grad = True
                bias.requires_grad = True

                # dimensions for mean and variance
                stats_dimensions = list(range(tensor.dim()))
                # perform on C dimension for tensor of shape (N, C, +)
                stats_dimensions.pop(1)

                running_mean = tensor.mean(stats_dimensions).detach()
                running_var = tensor.var(stats_dimensions).detach()
                enc_running_mean = encrypted_input.mean(stats_dimensions)
                enc_running_var = encrypted_input.var(stats_dimensions)

                reference = torch.nn.functional.batch_norm(
                    tensor, running_mean, running_var, weight=weight, bias=bias
                )

                encrypted_input = AutogradCrypTensor(encrypted_input)
                ctx = AutogradContext()
                batch_norm_fn = crypten.gradients.get_grad_fn("batchnorm")
                encrypted_out = batch_norm_fn.forward(
                    ctx,
                    (encrypted_input, weight, bias),
                    training=is_trainning,
                    running_mean=enc_running_mean,
                    running_var=enc_running_var,
                )

                # check forward
                self._check(
                    encrypted_out,
                    reference,
                    "batchnorm forward failed with trainning "
                    f"{is_trainning} on {tensor.dim()}-D",
                    tolerance=tolerance,
                )

                # check backward (input, weight, and bias gradients)
                reference.backward(reference)
                encrypted_grad = batch_norm_fn.backward(ctx, encrypted_out)
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
                        f"batchnorm backward {torch_gradient.name} failed"
                        f"with training {is_trainning} on {tensor.dim()}-D",
                        tolerance=tolerance,
                    )

    def test_cross_entropy(self):
        """Tests cross_entropy and binary_cross_entropy"""
        sizes = [(3, 2), (8, 4), (5, 10)]
        losses = ["binary_cross_entropy", "cross_entropy"]

        for size, loss in itertools.product(sizes, losses):
            batch_size, num_targets = size
            if loss == "binary_cross_entropy":
                tensor = get_random_test_tensor(
                    size=(batch_size,), max_value=1.0, is_float=True
                )
                tensor = tensor.abs().add_(0.001)

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
            tensor_encr = AutogradCrypTensor(
                crypten.cryptensor(tensor), requires_grad=True
            )
            reference = getattr(torch.nn.functional, loss)(tensor, target)
            out_encr = getattr(tensor_encr, loss)(target_encr)
            self._check(out_encr, reference, f"{loss} forward failed")

            # backward
            grad_out = get_random_test_tensor(size=reference.shape, is_float=True)
            grad_out_encr = crypten.cryptensor(grad_out)
            reference.backward(grad_out)
            out_encr.backward(grad_out_encr)
            self._check(tensor_encr.grad, tensor.grad, f"{loss} backward failed with")

    def test_view_reshape(self):
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

    def test_narrow_flatten(self):
        """Tests narrow and flatten gradients"""
        sizes = [(10,), (5, 4), (10, 6, 8)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("flatten", tensor)
            for dim in range(tensor.dim()):
                self._check_forward_backward("narrow", tensor, dim, 0, 2)
                self._check_forward_backward("narrow", tensor, dim, 1, 3)

    def test_flip(self):
        """Tests flip gradient"""
        sizes = [(2, 3, 7, 2), (5, 10, 15)]
        flips = [(0, 2, 1), (0, 1)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for flip in flips:
                self._check_forward_backward("flip", tensor, flip)

    def test_gather_scatter(self):
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
                tensor_encr = AutogradCrypTensor(
                    crypten.cryptensor(tensor), requires_grad=True
                )

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

    def test_take(self):
        """Tests take gradients"""
        sizes = [(10,), (5, 10), (2, 5, 10)]
        indices = [[0], [0, 5], [0, 2, 5, 8]]

        for size, index in itertools.product(sizes, indices):
            tensor = get_random_test_tensor(size=size, is_float=True)
            index = torch.tensor(index)
            self._check_forward_backward("take", tensor, index)

    def test_roll(self):
        """Tests roll gradients"""
        sizes = [(1, 10), (5, 10), (2, 5, 10)]
        shifts = [1, 3, (1, 2)]
        dims = [0, 1, (0, 1)]

        for size, shift_dim in itertools.product(sizes, zip(shifts, dims)):
            shift, dim = shift_dim
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("roll", tensor, shift, dim)

    def test_cumsum(self):
        """Tests cumsum gradient"""
        sizes = [(10,), (5, 10), (2, 5, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            for dim in range(tensor.dim()):
                self._check_forward_backward("cumsum", tensor, dim)

    def test_trace(self):
        """Tests trace gradient"""
        sizes = [(1, 1), (3, 3), (10, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("trace", tensor)

    def test_var(self):
        """Tests var gradient"""
        sizes = [(10,), (1, 10), (5, 10), (2, 5, 10)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("var", tensor)

    def test_getitem(self):
        """Tests getitem gradient"""
        sizes = [(10,), (10, 1), (5, 10), (5, 2, 10)]
        indices = [0, 1, 3]

        for size, index in itertools.product(sizes, indices):
            tensor = get_random_test_tensor(size=size, is_float=True)
            self._check_forward_backward("__getitem__", tensor, index)

    def test_pos_pow(self):
        """Test gradient crypten pos_pow"""
        for power in [3, -2, 1.75]:
            # ensure base is positive for pos_pow
            tensor = get_random_test_tensor(is_float=True, max_value=2) + 4
            tensor.requires_grad = True
            tensor_encr = AutogradCrypTensor(
                crypten.cryptensor(tensor), requires_grad=True
            )

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


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestGradients):
    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedFirstParty)
        super(TestTFP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestGradients):
    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)
        super(TestTTP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestGradients.benchmarks_enabled = True
    unittest.main()
