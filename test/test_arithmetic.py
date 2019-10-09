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
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.rng import generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.common.util import count_wraps
from crypten.mpc.primitives import ArithmeticSharedTensor


class TestArithmetic(MultiProcessTestCase):
    """
        This class tests all functions of the ArithmeticSharedTensor.
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
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

    def test_share_attr(self):
        """Tests share attribute getter and setter"""
        for is_float in (True, False):
            reference = get_random_test_tensor(is_float=is_float)
            encrypted_tensor = ArithmeticSharedTensor(reference)
            self.assertTrue(
                torch.equal(encrypted_tensor.share, encrypted_tensor.share),
                "share getter failed",
            )

            new_share = get_random_test_tensor(is_float=False)
            encrypted_tensor.share = new_share
            self.assertTrue(
                torch.equal(encrypted_tensor.share, new_share), "share setter failed"
            )

    def test_encrypt_decrypt(self):
        """
            Tests tensor encryption and decryption for both positive
            and negative values.
        """
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            reference = get_random_test_tensor(size=size, is_float=True)
            with self.benchmark(tensor_type="ArithmeticSharedTensor") as bench:
                for _ in bench.iters:
                    encrypted_tensor = ArithmeticSharedTensor(reference)
            self._check(encrypted_tensor, reference, "en/decryption failed")

    def test_arithmetic(self):
        """Tests arithmetic functions on encrypted tensor."""
        arithmetic_functions = ["add", "add_", "sub", "sub_", "mul", "mul_"]
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True)
                encrypted = ArithmeticSharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)
                private_type = tensor_type == ArithmeticSharedTensor
                self._check(
                    encrypted_out,
                    reference,
                    "%s %s failed" % ("private" if private_type else "public", func),
                )
                if "_" in func:
                    # Check in-place op worked
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
                            "private"
                            if tensor_type == ArithmeticSharedTensor
                            else "public",
                            func,
                        ),
                    )

                # Check encrypted vector with encrypted scalar works.
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True, size=(1,))
                encrypted1 = ArithmeticSharedTensor(tensor1)
                encrypted2 = ArithmeticSharedTensor(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted1, func)(encrypted2)
                self._check(encrypted_out, reference, "private %s failed" % func)

            tensor = get_random_test_tensor(is_float=True)
            reference = tensor * tensor
            encrypted = ArithmeticSharedTensor(tensor)
            encrypted_out = encrypted.square()
            self._check(encrypted_out, reference, "square failed")

        # Test radd, rsub, and rmul
        reference = 2 + tensor1
        encrypted = ArithmeticSharedTensor(tensor1)
        encrypted_out = 2 + encrypted
        self._check(encrypted_out, reference, "right add failed")

        reference = 2 - tensor1
        encrypted_out = 2 - encrypted
        self._check(encrypted_out, reference, "right sub failed")

        reference = 2 * tensor1
        encrypted_out = 2 * encrypted
        self._check(encrypted_out, reference, "right mul failed")

    def test_sum(self):
        tensor = get_random_test_tensor(size=(5, 100, 100), is_float=True)
        encrypted = ArithmeticSharedTensor(tensor)
        self._check(encrypted.sum(), tensor.sum(), "sum failed")

        for dim in [0, 1, 2]:
            reference = tensor.sum(dim)
            with self.benchmark(type="sum", dim=dim) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted.sum(dim)
            self._check(encrypted_out, reference, "sum failed")

    def test_div(self):
        """Tests division of encrypted tensor by scalar."""
        for function in ["div", "div_"]:
            for scalar in [2, 2.0]:
                tensor = get_random_test_tensor(is_float=True)

                reference = tensor.float().div(scalar)
                encrypted_tensor = ArithmeticSharedTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(scalar)
                self._check(encrypted_tensor, reference, "division failed")

                divisor = get_random_test_tensor(is_float=float)
                divisor += (divisor == 0).to(dtype=divisor.dtype)  # div by 0

                reference = tensor.div(divisor)
                encrypted_tensor = ArithmeticSharedTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(divisor)
                self._check(encrypted_tensor, reference, "division failed")

    def test_mean(self):
        """Tests computing means of encrypted tensors."""
        tensor = get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = ArithmeticSharedTensor(tensor)
        self._check(encrypted.mean(), tensor.mean(), "mean failed")

        for dim in [0, 1, 2]:
            reference = tensor.mean(dim)
            encrypted_out = encrypted.mean(dim)
            self._check(encrypted_out, reference, "mean failed")

    def test_wraps(self):
        num_parties = int(self.world_size)

        size = (5, 5)

        # Generate random sharing with internal value get_random_test_tensor()
        zero_shares = generate_random_ring_element((num_parties, *size))
        zero_shares = zero_shares - zero_shares.roll(1, dims=0)
        shares = list(zero_shares.unbind(0))
        shares[0] += get_random_test_tensor(size=size, is_float=False)

        # Note: This test relies on count_wraps function being correct
        reference = count_wraps(shares)

        # Sync shares between parties
        share = comm.get().scatter(shares, 0)

        encrypted_tensor = ArithmeticSharedTensor.from_shares(share)
        encrypted_wraps = encrypted_tensor.wraps()

        test_passed = (
            encrypted_wraps.reveal() == reference
        ).sum() == reference.nelement()
        self.assertTrue(test_passed, "%d-party wraps failed" % num_parties)

    def test_matmul(self):
        """Test matrix multiplication."""
        for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
            tensor = get_random_test_tensor(max_value=7, is_float=True)
            for width in range(2, tensor.nelement()):
                matrix_size = (tensor.nelement(), width)
                matrix = get_random_test_tensor(
                    max_value=7, size=matrix_size, is_float=True
                )
                reference = tensor.matmul(matrix)
                encrypted_tensor = ArithmeticSharedTensor(tensor)
                matrix = tensor_type(matrix)
                encrypted_tensor = encrypted_tensor.matmul(matrix)
                private_type = tensor_type == ArithmeticSharedTensor
                self._check(
                    encrypted_tensor,
                    reference,
                    "Private-%s matrix multiplication failed"
                    % ("private" if private_type else "public"),
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
                for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                    tensor1 = get_random_test_tensor(size=tensor_size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=tensor_size2, is_float=True)
                    encrypted = ArithmeticSharedTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)

                    reference = getattr(tensor1, func)(dimension, index, tensor2)
                    encrypted_out = getattr(encrypted, func)(
                        dimension, index, encrypted2
                    )
                    private = tensor_type == ArithmeticSharedTensor
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
                            "%s %s failed" % ("private" if private else "public", func),
                        )
                    else:
                        # Check original is not modified
                        self._check(
                            encrypted,
                            tensor1,
                            "%s %s failed" % ("private" if private else "public", func),
                        )

    def test_scatter_add(self):
        """Test index_add function of encrypted tensor"""
        funcs = ["scatter_add", "scatter_add_"]
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for func in funcs:
            for size in sizes:
                for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                    for dim in range(len(size)):
                        tensor1 = get_random_test_tensor(size=size, is_float=True)
                        tensor2 = get_random_test_tensor(size=size, is_float=True)
                        index = get_random_test_tensor(size=size, is_float=False)
                        index = index.abs().clamp(0, 4)
                        encrypted = ArithmeticSharedTensor(tensor1)
                        encrypted2 = tensor_type(tensor2)
                        reference = getattr(tensor1, func)(dim, index, tensor2)
                        encrypted_out = getattr(encrypted, func)(dim, index, encrypted2)
                        private = tensor_type == ArithmeticSharedTensor
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

    def test_dot_ger(self):
        """Test dot product of vector and encrypted tensor."""
        for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
            tensor1 = get_random_test_tensor(is_float=True).squeeze()
            tensor2 = get_random_test_tensor(is_float=True).squeeze()
            dot_reference = tensor1.dot(tensor2)
            ger_reference = torch.ger(tensor1, tensor2)

            tensor2 = tensor_type(tensor2)

            # dot
            encrypted_tensor = ArithmeticSharedTensor(tensor1)
            encrypted_out = encrypted_tensor.dot(tensor2)
            self._check(
                encrypted_out,
                dot_reference,
                "%s dot product failed" % "private"
                if tensor_type == ArithmeticSharedTensor
                else "public",
            )

            # ger
            encrypted_tensor = ArithmeticSharedTensor(tensor1)
            encrypted_out = encrypted_tensor.ger(tensor2)
            self._check(
                encrypted_out,
                ger_reference,
                "%s outer product failed" % "private"
                if tensor_type == ArithmeticSharedTensor
                else "public",
            )

    def test_squeeze(self):
        tensor = get_random_test_tensor(is_float=True)

        for dim in [0, 1, 2]:
            # Test unsqueeze
            reference = tensor.unsqueeze(dim)

            encrypted = ArithmeticSharedTensor(tensor)
            with self.benchmark(type="unsqueeze", dim=dim) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted.unsqueeze(dim)
            self._check(encrypted_out, reference, "unsqueeze failed")

            # Test squeeze
            encrypted = ArithmeticSharedTensor(tensor.unsqueeze(0))
            with self.benchmark(type="squeeze", dim=dim) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted.squeeze()
            self._check(encrypted_out, reference.squeeze(), "squeeze failed")

            # Check that the encrypted_out and encrypted point to the same
            # thing.
            encrypted_out[0:2] = torch.FloatTensor([0, 1])
            ref = encrypted.squeeze().get_plain_text()
            self._check(encrypted_out, ref, "squeeze failed")

    def test_transpose(self):
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = ArithmeticSharedTensor(tensor)

            if len(size) == 2:  # t() asserts dim == 2
                reference = tensor.t()
                with self.benchmark(niters=10) as bench:
                    for _ in bench.iters:
                        encrypted_out = encrypted_tensor.t()
                self._check(encrypted_out, reference, "t() failed")

            for dim0 in range(len(size)):
                for dim1 in range(len(size)):
                    reference = tensor.transpose(dim0, dim1)
                    with self.benchmark(niters=10) as bench:
                        for _ in bench.iters:
                            encrypted_out = encrypted_tensor.transpose(dim0, dim1)
                    self._check(encrypted_out, reference, "transpose failed")

    def test_conv(self):
        """Test convolution of encrypted tensor with public/private tensors."""
        for kernel_type in [lambda x: x, ArithmeticSharedTensor]:
            for matrix_width in range(2, 5):
                for kernel_width in range(1, matrix_width):
                    for padding in range(kernel_width // 2 + 1):
                        matrix_size = (5, matrix_width)
                        matrix = get_random_test_tensor(size=matrix_size, is_float=True)

                        kernel_size = (kernel_width, kernel_width)
                        kernel = get_random_test_tensor(size=kernel_size, is_float=True)

                        matrix = matrix.unsqueeze(0).unsqueeze(0)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)

                        reference = F.conv2d(matrix, kernel, padding=padding)
                        encrypted_matrix = ArithmeticSharedTensor(matrix)
                        encrypted_kernel = kernel_type(kernel)
                        with self.benchmark(
                            kernel_type=kernel_type.__name__, matrix_width=matrix_width
                        ) as bench:
                            for _ in bench.iters:
                                encrypted_conv = encrypted_matrix.conv2d(
                                    encrypted_kernel, padding=padding
                                )

                        self._check(encrypted_conv, reference, "conv2d failed")

    def test_pooling(self):
        """Test avgPool of encrypted tensor."""
        for func in ["avg_pool2d", "sum_pool2d"]:
            for width in range(2, 5):
                for width2 in range(1, width):
                    matrix_size = (1, 4, 5, width)
                    matrix = get_random_test_tensor(size=matrix_size, is_float=True)
                    pool_size = width2
                    for stride in range(1, width2):
                        for padding in range(2):
                            reference = F.avg_pool2d(
                                matrix, pool_size, stride=stride, padding=padding
                            )
                            if func == "sum_pool2d":
                                reference *= width2 * width2

                            encrypted_matrix = ArithmeticSharedTensor(matrix)
                            with self.benchmark(func=func, width=width) as bench:
                                for _ in bench.iters:
                                    encrypted_pool = getattr(encrypted_matrix, func)(
                                        pool_size, stride=stride, padding=padding
                                    )
                            self._check(encrypted_pool, reference, "%s failed" % func)

    def test_take(self):
        """Tests take function of encrypted tensor"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor([[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long)
        tensor = get_random_test_tensor(size=tensor_size, is_float=True)

        # Test when dimension!=None
        for dimension in range(0, 4):
            reference = torch.from_numpy(tensor.numpy().take(index, dimension))
            encrypted_tensor = ArithmeticSharedTensor(tensor)
            encrypted_out = encrypted_tensor.take(index, dimension)
            self._check(encrypted_out, reference, "take function failed: dimension set")

        # Test when dimension is default (i.e. None)
        sizes = [(15,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = ArithmeticSharedTensor(tensor)
            take_indices = [[0], [10], [0, 5, 10]]
            for indices in take_indices:
                indices = torch.tensor(indices)
                self._check(
                    encrypted_tensor.take(indices),
                    tensor.take(indices),
                    f"take failed with indices {indices}",
                )

    def test_get_set(self):
        for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = get_random_test_tensor(size=(size, size), is_float=True)
                reference = tensor[:, 0]

                encrypted_tensor = ArithmeticSharedTensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,), is_float=True)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = ArithmeticSharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = ArithmeticSharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

    def test_pad(self):
        sizes = [(1,), (5,), (1, 1), (5, 5), (5, 5, 5), (5, 3, 32, 32)]
        pads = [
            (0, 0, 0, 0),
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
            encrypted_tensor = ArithmeticSharedTensor(tensor)

            for pad in pads:
                for value in [0, 1, 10]:
                    for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                        if tensor.dim() < 2:
                            pad = pad[:2]
                        reference = torch.nn.functional.pad(tensor, pad, value=value)
                        encrypted_value = tensor_type(value)
                        with self.benchmark(tensor_type=tensor_type.__name__) as bench:
                            for _ in bench.iters:
                                encrypted_out = encrypted_tensor.pad(
                                    pad, value=encrypted_value
                                )
                        self._check(encrypted_out, reference, "pad failed")

    def test_broadcast(self):
        """Test broadcast functionality."""
        arithmetic_functions = ["add", "sub", "mul", "div"]
        arithmetic_sizes = [
            (),
            (1,),
            (2,),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 2),
            (1, 1, 1, 1),
            (1, 1, 1, 2),
            (1, 1, 2, 1),
            (1, 2, 1, 1),
            (2, 1, 1, 1),
            (2, 2, 2, 2),
        ]
        matmul_sizes = [(1, 1), (1, 5), (5, 1), (5, 5)]
        batch_dims = [(), (1,), (5,), (1, 1), (1, 5), (5, 5)]

        for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
            for func in arithmetic_functions:
                for size1, size2 in itertools.combinations(arithmetic_sizes, 2):
                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=size2, is_float=True)

                    # ArithmeticSharedTensors can't divide by negative
                    # private values - MPCTensor overrides this to allow negatives
                    # multiply denom by 10 to avoid division by small num
                    if func == "div" and tensor_type == ArithmeticSharedTensor:
                        continue

                    encrypted1 = ArithmeticSharedTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(tensor2)
                    encrypted_out = getattr(encrypted1, func)(encrypted2)

                    private = isinstance(encrypted2, ArithmeticSharedTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s %s broadcast failed"
                        % ("private" if private else "public", func),
                    )

            for size in matmul_sizes:
                for batch1, batch2 in itertools.combinations(batch_dims, 2):
                    size1 = (*batch1, *size)
                    size2 = (*batch2, *size)

                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=size2, is_float=True)
                    tensor2 = tensor1.transpose(-2, -1)

                    encrypted1 = ArithmeticSharedTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)

                    reference = tensor1.matmul(tensor2)
                    encrypted_out = encrypted1.matmul(encrypted2)
                    private = isinstance(encrypted2, ArithmeticSharedTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s matmul broadcast failed"
                        % ("private" if private else "public"),
                    )

    def test_inplace(self):
        """Test inplace vs. out-of-place functions"""
        for op in ["add", "sub", "mul", "div"]:
            for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True)

                # ArithmeticSharedTensors can't divide by negative
                # private values - MPCTensor overrides this to allow negatives
                if op == "div" and tensor_type == ArithmeticSharedTensor:
                    continue

                reference = getattr(torch, op)(tensor1, tensor2)

                encrypted1 = ArithmeticSharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                input_plain_id = id(encrypted1.share)
                input_encrypted_id = id(encrypted1)

                # Test that out-of-place functions do not modify the input
                private = isinstance(encrypted2, ArithmeticSharedTensor)
                encrypted_out = getattr(encrypted1, op)(encrypted2)
                self._check(
                    encrypted1,
                    tensor1,
                    "%s out-of-place %s modifies input"
                    % ("private" if private else "public", op),
                )
                self._check(
                    encrypted_out,
                    reference,
                    "%s out-of-place %s produces incorrect output"
                    % ("private" if private else "public", op),
                )
                self.assertFalse(id(encrypted_out.share) == input_plain_id)
                self.assertFalse(id(encrypted_out) == input_encrypted_id)

                # Test that in-place functions modify the input
                encrypted_out = getattr(encrypted1, op + "_")(encrypted2)
                self._check(
                    encrypted1,
                    reference,
                    "%s in-place %s_ does not modify input"
                    % ("private" if private else "public", op),
                )
                self._check(
                    encrypted_out,
                    reference,
                    "%s in-place %s_ produces incorrect output"
                    % ("private" if private else "public", op),
                )
                self.assertTrue(id(encrypted_out.share) == input_plain_id)
                self.assertTrue(id(encrypted_out) == input_encrypted_id)

    def test_control_flow_failure(self):
        """Tests that control flow fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        encrypted_tensor = ArithmeticSharedTensor(tensor)
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

    def test_src_failure(self):
        """Tests that out-of-bounds src fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        for src in [None, "abc", -2, self.world_size]:
            with self.assertRaises(AssertionError):
                ArithmeticSharedTensor(tensor, src=src)

    def test_src_match_input_data(self):
        """Tests incorrect src in ArithmeticSharedTensor fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        tensor.src = 0
        for testing_src in [None, "abc", -2, self.world_size]:
            with self.assertRaises(AssertionError):
                ArithmeticSharedTensor(tensor, src=testing_src)

    def test_where(self):
        """Tests where() conditional element selection"""
        sizes = [(10,), (5, 10), (1, 5, 10)]
        y_types = [lambda x: x, ArithmeticSharedTensor]

        for size, y_type in itertools.product(sizes, y_types):
            tensor1 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = ArithmeticSharedTensor(tensor1)
            tensor2 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                get_random_test_tensor(max_value=1, size=size, is_float=False) + 1
            )
            condition_encrypted = ArithmeticSharedTensor(condition_tensor)
            condition_bool = condition_tensor.bool()

            reference_out = tensor1.where(condition_bool, tensor2)

            encrypted_out = encrypted_tensor1.where(condition_bool, encrypted_tensor2)
            y_is_private = y_type == ArithmeticSharedTensor
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

    # TODO: Write the following unit tests
    @unittest.skip("Test not implemented")
    def test_gather_scatter(self):
        pass

    @unittest.skip("Test not implemented")
    def test_split(self):
        pass


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestArithmetic.benchmarks_enabled = True
    unittest.main()
