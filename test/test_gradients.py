#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import torch
import torch.nn.functional as F
from crypten.autograd_cryptensor import AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor
from crypten.mpc import MPCTensor


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


class TestGradients(MultiProcessTestCase):
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

    def _check_forward_backward(self, fn_name, input_tensor, *args, msg=None, **kwargs):
        if msg is None:
            msg = f"{fn_name} grad_fn incorrect"

        for requires_grad in [True]:
            # Setup input
            input = input_tensor.clone()
            input.requires_grad = requires_grad
            input_encr = AutogradCrypTensor(
                crypten.cryptensor(input), requires_grad=requires_grad
            )

            for private in [False, True]:
                input.grad = None
                input_encr.grad = None

                # Setup args
                args_encr = list(args)
                for i, arg in enumerate(args):
                    if private and is_float_tensor(arg):
                        args_encr[i] = AutogradCrypTensor(
                            crypten.cryptensor(arg), requires_grad=requires_grad
                        )
                        args_encr[i].grad = None  # zero grad
                    if is_float_tensor(arg):
                        args[i].requires_grad = requires_grad
                        args[i].grad = None  # zero grad

                # Check forward pass
                if hasattr(input, fn_name):
                    reference = getattr(input, fn_name)(*args, **kwargs)
                elif hasattr(F, fn_name):
                    reference = getattr(F, fn_name)(input, *args, **kwargs)
                elif fn_name == "square":
                    reference = input.pow(2)
                else:
                    raise ValueError("unknown PyTorch function: %s" % fn_name)

                encrypted_out = getattr(input_encr, fn_name)(*args_encr, **kwargs)

                # Remove argmax output from max / min
                if isinstance(encrypted_out, (list, tuple)):
                    reference = reference[0]
                    encrypted_out = encrypted_out[0]

                self._check(encrypted_out, reference, msg + " in forward")

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
                encrypted_out.backward(grad_output_encr)

                self._check(input_encr.grad, input.grad, msg + " in backward")
                for i, arg_encr in enumerate(args_encr):
                    if crypten.is_encrypted_tensor(arg_encr):
                        self._check(
                            arg_encr.grad, args[i].grad, msg + " in backward args"
                        )

    def test_arithmetic(self):
        """Tests arithmetic functions with broadcasting."""
        arithmetic_functions = ["add", "sub", "mul", "div"]
        for func in arithmetic_functions:
            # Test on operator
            ofunc = "truediv" if func == "div" else func
            ofunc = "__" + ofunc + "__"

            # Test both left functions and right functions
            rfunc = ofunc[:2] + "r" + ofunc[2:]

            # Test on both float inputs and tensor inputs
            for use_tensor in [False, True]:
                for size1 in SIZES:
                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    if use_tensor:
                        for size2 in SIZES:
                            tensor2 = get_random_test_tensor(
                                min_value=0.5, size=size2, is_float=True
                            )  # do not divide by value very close to zero
                            self._check_forward_backward(func, tensor1, tensor2)
                            self._check_forward_backward(ofunc, tensor1, tensor2)
                            self._check_forward_backward(rfunc, tensor1, tensor2)
                    else:
                        scalar = 2.0
                        self._check_forward_backward(func, tensor1, scalar)
                        self._check_forward_backward(ofunc, tensor1, scalar)
                        self._check_forward_backward(rfunc, tensor1, scalar)

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

        for size in matmul_sizes:
            for batch1, batch2 in itertools.combinations(batch_dims, 2):
                size1 = (*batch1, *size)
                size2 = (*batch2, *size)

                tensor1 = get_random_test_tensor(size=size1, is_float=True)
                tensor2 = get_random_test_tensor(size=size2, is_float=True)
                tensor2 = tensor2.transpose(-2, -1)

                self._check_forward_backward("matmul", tensor1, tensor2)

    def test_unary_functions(self):
        """Test unary functions on tensors of various sizes."""
        unary_functions = [
            "neg",
            "__neg__",
            "exp",
            "reciprocal",
            "abs",
            "sign",
            "relu",
            "sin",
            "cos",
            "square",
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

        for size in SIZES:
            tensor = get_random_test_tensor(size=size, is_float=True)

            # Check dim 0 if tensor is 0-dimensional
            dims = 1 if tensor.dim() == 0 else tensor.dim()
            for dim in range(dims):
                self._check_forward_backward("softmax", tensor, dim)

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

        kernel_sizes = [(1, 1), (2, 2), (5, 5), (2, 3)]
        paddings = [0, 1, (0, 1)]
        # TODO: Fix conv2d backward with stride = 2
        # strides = [1, 2, (1, 2)]
        strides = [1]
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

    def test_pooling(self):
        """Test pooling functions on encrypted tensor"""
        image_sizes = [(5, 5), (16, 7)]
        nchannels = [1, 3, 5]
        nbatches = [1, 3, 5]

        kernel_sizes = [1, 2]
        paddings = [0, 1]
        strides = [1, 2]

        # TODO: Fix the following cases:
        #
        # 1) kernel_sizes / paddings / strides with tuples of uneven size (e.g. (1, 2))
        # 2) Correct avg_pool2d backward for cases listed in its backward TODO

        funcs = ["avg_pool2d", "max_pool2d"]
        for image_size, channels, batches, kernel_size in itertools.product(
            image_sizes, nchannels, nbatches, kernel_sizes
        ):
            size = (batches, channels, *image_size)
            image = get_random_test_tensor(size=size, is_float=True)

            for padding, stride, func in itertools.product(paddings, strides, funcs):
                # TODO: Correct avg_pool2d backward:
                if size[1] > 1 and func == "avg_pool2d":
                    continue

                # Skip invalid padding sizes
                if kernel_size == 1 and padding == 1:
                    continue
                self._check_forward_backward(
                    func, image, kernel_size, padding=padding, stride=stride
                )

    def test_pow(self):
        """Tests pow function"""
        for size in SIZES:
            tensor = get_random_test_tensor(
                size=size, min_value=0.5, is_float=True
            )  # prevent division by values close to zero
            for power in [-3, -2, -1, 0, 1, 2, 3]:
                self._check_forward_backward("pow", tensor, power)
                self._check_forward_backward("pow", tensor, float(power))

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

    '''
    def test_var(self):
        """Tests computing variances of encrypted tensors."""
        tensor = get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = crypten.cryptensor(tensor)
        self._check(encrypted.var(), tensor.var(), "var failed")

        for dim in [0, 1, 2]:
            reference = tensor.var(dim)
            encrypted_out = encrypted.var(dim)
            self._check(encrypted_out, reference, "var failed")

    def test_take(self):
        """Tests take function on encrypted tensor"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor([[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long)
        tensor = get_random_test_tensor(size=tensor_size, is_float=True)

        # Test when dimension!=None
        for dimension in range(0, 4):
            reference = torch.from_numpy(tensor.numpy().take(index, dimension))
            encrypted_tensor = crypten.cryptensor(tensor)
            encrypted_out = encrypted_tensor.take(index, dimension)
            self._check(encrypted_out, reference, "take function failed: dimension set")

        # Test when dimension is default (i.e. None)
        sizes = [(15,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)
            take_indices = [[0], [10], [0, 5, 10]]
            for indices in take_indices:
                indices = torch.tensor(indices)
                self._check(
                    encrypted_tensor.take(indices),
                    tensor.take(indices),
                    f"take failed with indices {indices}",
                )

    def test_approximations(self):
        """Test appoximate functions (exp, log, sqrt, reciprocal, pos_pow)"""

        def test_with_inputs(func, input):
            encrypted_tensor = crypten.cryptensor(input)
            reference = getattr(tensor, func)()
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

        # Test on [-10, 10] range
        full_range_cases = ["exp"]
        tensor = torch.tensor([0.01 * i for i in range(-1000, 1001, 1)])
        for func in full_range_cases:
            test_with_inputs(func, tensor)

        # Test on [0, 10] range
        tensor[tensor == 0] = 1.0
        non_zero_cases = ["reciprocal"]
        for func in non_zero_cases:
            test_with_inputs(func, tensor)

        # Test on [0, 10] range
        tensor = tensor[1001:]
        pos_cases = ["log", "sqrt"]
        for func in pos_cases:
            test_with_inputs(func, tensor)

        # Test pos_pow with several exponents
        encrypted_tensor = crypten.cryptensor(tensor)

        # Reduced the max_value so approximations have less absolute error
        tensor_exponent = get_random_test_tensor(
            max_value=2, size=tensor.size(), is_float=True
        )
        exponents = [-3, -2, -1, 0, 1, 2, 3, tensor_exponent]
        exponents += [crypten.cryptensor(tensor_exponent)]
        for p in exponents:
            if isinstance(p, MPCTensor):
                reference = tensor.pow(p.get_plain_text())
            else:
                reference = tensor.pow(p)
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor.pos_pow(p)
            self._check(encrypted_out, reference, f"pos_pow failed with power {p}")

    def test_get_set(self):
        """Tests element setting and getting by index"""
        for tensor_type in [lambda x: x, MPCTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = get_random_test_tensor(size=(size, size), is_float=True)
                reference = tensor[:, 0]

                encrypted_tensor = crypten.cryptensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,), is_float=True)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = crypten.cryptensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = crypten.cryptensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

    def test_index_add(self):
        """Test index_add function of encrypted tensor"""
        index_add_functions = ["index_add", "index_add_"]
        tensor_size1 = [5, 5, 5, 5]
        index = torch.tensor([1, 2, 3, 4, 4, 2, 1, 3], dtype=torch.long)
        for dimension in range(0, 4):
            tensor_size2 = [5, 5, 5, 5]
            tensor_size2[dimension] = index.size(0)
            for func in index_add_functions:
                for tensor_type in [lambda x: x, MPCTensor]:
                    tensor1 = get_random_test_tensor(size=tensor_size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=tensor_size2, is_float=True)
                    encrypted = crypten.cryptensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(dimension, index, tensor2)
                    encrypted_out = getattr(encrypted, func)(
                        dimension, index, encrypted2
                    )
                    private_type = tensor_type == MPCTensor
                    self._check(
                        encrypted_out,
                        reference,
                        "%s %s failed"
                        % ("private" if private_type else "public", func),
                    )
                    if func.endswith("_"):
                        # Check in-place index_add worked
                        self._check(
                            encrypted,
                            reference,
                            "%s %s failed"
                            % ("private" if private_type else "public", func),
                        )
                    else:
                        # Check original is not modified
                        self._check(
                            encrypted,
                            tensor1,
                            "%s %s failed"
                            % (
                                "private" if tensor_type == MPCTensor else "public",
                                func,
                            ),
                        )

    def test_scatter_add(self):
        """Test index_add function of encrypted tensor"""
        funcs = ["scatter_add", "scatter_add_"]
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for func in funcs:
            for size in sizes:
                for tensor_type in [lambda x: x, MPCTensor]:
                    for dim in range(len(size)):
                        tensor1 = get_random_test_tensor(size=size, is_float=True)
                        tensor2 = get_random_test_tensor(size=size, is_float=True)
                        index = get_random_test_tensor(size=size, is_float=False)
                        index = index.abs().clamp(0, 4)
                        encrypted = crypten.cryptensor(tensor1)
                        encrypted2 = tensor_type(tensor2)
                        reference = getattr(tensor1, func)(dim, index, tensor2)
                        encrypted_out = getattr(encrypted, func)(dim, index, encrypted2)
                        private = tensor_type == MPCTensor
                        self._check(
                            encrypted_out,
                            reference,
                            "%s %s failed" % ("private" if private else "public", func),
                        )
                        if func.endswith("_"):
                            # Check in-place index_add worked
                            self._check(
                                encrypted,
                                reference,
                                "%s %s failed"
                                % ("private" if private else "public", func),
                            )
                        else:
                            # Check original is not modified
                            self._check(
                                encrypted,
                                tensor1,
                                "%s %s failed"
                                % ("private" if private else "public", func),
                            )

    def test_index_select(self):
        """Tests index_select of encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)
            indices = [[0], [0, 3], [0, 2, 4]]

            for dim in range(tensor.dim()):
                for index in indices:
                    index_tensor = torch.tensor(index, dtype=torch.long)
                    reference = tensor.index_select(dim, index_tensor)
                    encrypted_out = encrypted_tensor.index_select(dim, index_tensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "index_select failed at dim {dim} and index {index}",
                    )

    def test_narrow(self):
        """Tests narrow function."""
        sizes = [(5, 6), (5, 6, 7), (6, 7, 8, 9)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encr_tensor = crypten.cryptensor(tensor)
            for dim in range(len(size)):
                for start in range(size[dim] - 2):
                    for length in range(1, size[dim] - start):
                        tensor_narrow = tensor.narrow(dim, start, length)
                        encr_tensor_narrow = encr_tensor.narrow(dim, start, length)
                        self._check(
                            encr_tensor_narrow,
                            tensor_narrow,
                            "narrow failed along dimension %d" % dim,
                        )

    def test_repeat_expand(self):
        """Tests repeat and expand of encrypted tensors."""
        sizes = [(1, 8), (4, 1, 8)]
        repeat_dims = [(4, 2, 1), (4, 2, 10)]
        expand_dims = [(4, 2, 8), (4, 5, 8), (10, 4, 5, 8)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            for dims in repeat_dims:
                encrypted_tensor_repeated = encrypted_tensor.repeat(*dims)
                # test that repeat copies tensor's data
                self.assertNotEqual(
                    id(encrypted_tensor_repeated._tensor), id(encrypted_tensor._tensor)
                )
                self._check(
                    encrypted_tensor_repeated,
                    tensor.repeat(*dims),
                    f"repeat failed with dims {dims}",
                )

            for dims in expand_dims:
                encrypted_tensor_expanded = encrypted_tensor.expand(*dims)
                # test that expand creates a view into the same underlying tensor
                self.assertNotEqual(
                    id(encrypted_tensor_expanded.share), id(encrypted_tensor.share)
                )
                self._check(
                    encrypted_tensor_expanded,
                    tensor.expand(*dims),
                    f"repeat failed with dims {dims}",
                )

    def test_view_flatten(self):
        """Tests view and flatten of encrypted tensors."""
        sizes = [(100,), (4, 25), (2, 5, 10)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)
            for dim in range(tensor.dim()):
                self._check(
                    encrypted_tensor.flatten(start_dim=dim),
                    tensor.flatten(start_dim=dim),
                    f"flatten failed with dim {dim}",
                )

            shapes = [(100), (5, 20), (10, 2, 5), (-1, 10)]
            for shape in shapes:
                self._check(
                    encrypted_tensor.view(shape),
                    tensor.view(shape),
                    f"view failed with shape {shape}",
                )

    def test_roll(self):
        """Tests roll of encrypted tensors."""
        sizes = [(10, 1), (5, 2), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)
            roll_shifts = [1, 2, 3, (2, 1)]
            roll_dims = [0, 1, 0, (0, 1)]

            for shifts, dims in zip(roll_shifts, roll_dims):
                encrypted_tensor_rolled = encrypted_tensor.roll(shifts, dims=dims)
                self.assertEqual(encrypted_tensor_rolled.numel(), tensor.numel())
                self._check(
                    encrypted_tensor_rolled,
                    tensor.roll(shifts, dims=dims),
                    f"roll failed with shift {shifts} and dims {dims}",
                )

    def test_unfold(self):
        """Tests unfold of encrypted tensors."""
        tensor_sizes = [(8,), (15, 10, 5), (5, 10, 15, 20)]
        for tensor_size in tensor_sizes:
            tensor = get_random_test_tensor(size=tensor_size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            for size, step in itertools.product(range(1, 4), range(1, 4)):
                # check unfold along higher dimension if possible
                for dim in range(tensor.dim()):
                    self._check(
                        encrypted_tensor.unfold(dim, size, step),
                        tensor.unfold(dim, size, step),
                        "unfold failed with dim "
                        f"{dim}, size {size}, and step {step}",
                    )

    def test_to(self):
        """Tests Arithemetic/Binary SharedTensor type conversions."""
        from crypten.mpc.ptype import ptype as Ptype

        tensor = get_random_test_tensor(is_float=True)
        encrypted_tensor = crypten.cryptensor(tensor)
        self.assertEqual(encrypted_tensor.ptype, Ptype.arithmetic)

        binary_encrypted_tensor = encrypted_tensor.to(Ptype.binary)
        self.assertEqual(binary_encrypted_tensor.ptype, Ptype.binary)

        # check original encrypted_tensor was not modified after conversion
        self._check(
            encrypted_tensor,
            tensor,
            "encrypted_tensor was modified during conversion to BinarySharedTensor.",
        )

        encrypted_from_binary = binary_encrypted_tensor.to(Ptype.arithmetic)
        self._check(
            encrypted_from_binary,
            tensor,
            "to failed from BinarySharedTensor to ArithmeticSharedTensor",
        )

    def test_cumsum(self):
        """Tests cumulative sum on encrypted tensors."""
        sizes = [(8,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            for dim in range(tensor.dim()):
                self._check(
                    encrypted_tensor.cumsum(dim),
                    tensor.cumsum(dim),
                    f"cumsum failed along {dim} dim",
                )

    def test_trace(self):
        """Tests trace operation on 2D encrypted tensors."""
        sizes = [(3, 3), (10, 10), (2, 3)]

        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            self._check(encrypted_tensor.trace(), tensor.trace(), "trace failed")

    def test_flip(self):
        """Tests flip operation on encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = crypten.cryptensor(tensor)

            flip_dims = [(0,), (0, 1), (0, 1, 2)]

            for dims in flip_dims:
                if len(dims) <= tensor.dim():
                    self._check(
                        encrypted_tensor.flip(dims),
                        tensor.flip(dims),
                        f"flip failed with {dims} dims",
                    )

    def test_control_flow_failure(self):
        """Tests that control flow fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        encrypted_tensor = crypten.cryptensor(tensor)
        with self.assertRaises(RuntimeError):
            if encrypted_tensor:
                pass

        with self.assertRaises(RuntimeError):
            tensor = 5 if encrypted_tensor else 0

        with self.assertRaises(RuntimeError):
            if False:
                pass
            elif encrypted_tensor:
                pass

    def test_where(self):
        """Tests where() conditional element selection"""
        sizes = [(10,), (5, 10), (1, 5, 10)]
        y_types = [lambda x: x, MPCTensor]

        for size, y_type in itertools.product(sizes, y_types):
            tensor1 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = crypten.cryptensor(tensor1)
            tensor2 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                get_random_test_tensor(max_value=1, size=size, is_float=False) + 1
            )
            condition_encrypted = crypten.cryptensor(condition_tensor)
            condition_bool = condition_tensor.bool()

            reference_out = tensor1.where(condition_bool, tensor2)

            encrypted_out = encrypted_tensor1.where(condition_bool, encrypted_tensor2)
            y_is_private = y_type == MPCTensor
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with public condition",
            )

            encrypted_out = encrypted_tensor1.where(
                condition_encrypted, encrypted_tensor2
            )
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with private condition",
            )

            # test scalar y
            scalar = get_random_test_tensor(max_value=0, size=[1], is_float=True)
            self._check(
                encrypted_tensor1.where(condition_bool, scalar),
                tensor1.where(condition_bool, scalar),
                "where failed against scalar y with public condition",
            )

            self._check(
                encrypted_tensor1.where(condition_encrypted, scalar),
                tensor1.where(condition_bool, scalar),
                "where failed against scalar y with private condition",
            )

    def test_batchnorm(self):
        """Tests batchnorm"""
        sizes = [(2, 3, 10), (2, 3, 5, 10), (2, 3, 5, 10, 15)]

        for size in sizes:
            dim = len(size)
            # set max_value to 1 to avoid numerical precision in var division
            tensor = get_random_test_tensor(size=size, max_value=1, is_float=True)
            encrypted = crypten.cryptensor(tensor)

            weight = get_random_test_tensor(size=[3], max_value=1, is_float=True)
            bias = get_random_test_tensor(size=[3], max_value=1, is_float=True)

            # dimensions for mean and variance
            dimensions = list(range(dim))
            dimensions.pop(1)
            running_mean = tensor.mean(dimensions)
            running_var = tensor.var(dimensions)

            reference = torch.nn.functional.batch_norm(
                tensor, running_mean, running_var, weight=weight, bias=bias
            )

            # training false with given running mean and var
            encrypted_out = encrypted.batchnorm(
                None,
                weight,
                bias,
                training=False,
                running_mean=running_mean,
                running_var=running_var,
            )
            self._check(
                encrypted_out,
                reference,
                f"batchnorm failed with train False and dim {dim}",
            )

            # training true
            encrypted_out = encrypted.batchnorm(None, weight, bias, training=True)
            self._check(
                encrypted_out,
                reference,
                f"batchnorm failed with train True and dim {dim}",
            )

    # TODO: Write the following unit tests
    @unittest.skip("Test not implemented")
    def test_gather_scatter(self):
        pass
    '''


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestGradients.benchmarks_enabled = True
    unittest.main()
