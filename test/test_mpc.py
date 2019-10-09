#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import math
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor
from test.multithread_test_case import MultiThreadTestCase

import torch
import torch.nn.functional as F
from crypten.common.tensor_types import is_float_tensor
from crypten.mpc import MPCTensor, ptype


class TestMPC(MultiProcessTestCase):
    """
        This class tests all functions of MPCTensor.
    """

    benchmarks_enabled = False

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

    def _check_tuple(self, encrypted_tuple, reference, msg, tolerance=None):
        self.assertTrue(isinstance(encrypted_tuple, tuple))
        self.assertEqual(len(encrypted_tuple), len(reference))
        for i in range(len(reference)):
            self._check(encrypted_tuple[i], reference[i], msg, tolerance=tolerance)

    def test_repr(self):
        a = get_random_test_tensor(size=(1,))
        arithmetic = MPCTensor(a, ptype=ptype.arithmetic)
        binary = MPCTensor(a, ptype=ptype.binary)

        # Make sure these don't crash
        print(arithmetic)
        repr(arithmetic)

        print(binary)
        repr(binary)

    def test_share_attr(self):
        """Tests share attribute getter and setter"""
        for is_float in (True, False):
            reference = get_random_test_tensor(is_float=is_float)
            encrypted_tensor = MPCTensor(reference)
            underlying_tensor = encrypted_tensor.share
            self.assertTrue(
                torch.equal(encrypted_tensor.share, underlying_tensor),
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
            with self.benchmark(tensor_type="MPCTensor") as bench:
                for _ in bench.iters:
                    encrypted_tensor = MPCTensor(reference)
                    self._check(encrypted_tensor, reference, "en/decryption failed")
                    encrypted_tensor2 = encrypted_tensor.new(reference)
                    self.assertIsInstance(
                        encrypted_tensor2, MPCTensor, "new() returns incorrect type"
                    )
                    self._check(encrypted_tensor2, reference, "en/decryption failed")

    def test_arithmetic(self):
        """Tests arithmetic functions on encrypted tensor."""
        arithmetic_functions = ["add", "add_", "sub", "sub_", "mul", "mul_"]
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True)
                encrypted = MPCTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)
                self._check(
                    encrypted_out,
                    reference,
                    "%s %s failed"
                    % ("private" if tensor_type == MPCTensor else "public", func),
                )
                if "_" in func:
                    # Check in-place op worked
                    self._check(
                        encrypted,
                        reference,
                        "%s %s failed"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )
                else:
                    # Check original is not modified
                    self._check(
                        encrypted,
                        tensor1,
                        "%s %s failed"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )

                # Check encrypted vector with encrypted scalar works.
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True, size=(1,))
                encrypted1 = MPCTensor(tensor1)
                encrypted2 = MPCTensor(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted1, func)(encrypted2)
                self._check(encrypted_out, reference, "private %s failed" % func)

            tensor = get_random_test_tensor(is_float=True)
            reference = tensor * tensor
            encrypted = MPCTensor(tensor)
            encrypted_out = encrypted.square()
            self._check(encrypted_out, reference, "square failed")

        # Test radd, rsub, and rmul
        reference = 2 + tensor1
        encrypted = MPCTensor(tensor1)
        encrypted_out = 2 + encrypted
        self._check(encrypted_out, reference, "right add failed")

        reference = 2 - tensor1
        encrypted_out = 2 - encrypted
        self._check(encrypted_out, reference, "right sub failed")

        reference = 2 * tensor1
        encrypted_out = 2 * encrypted
        self._check(encrypted_out, reference, "right mul failed")

    def test_sum(self):
        """Tests sum reduction on encrypted tensor."""
        tensor = get_random_test_tensor(size=(100, 100), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.sum(), tensor.sum(), "sum failed")

        for dim in [0, 1]:
            reference = tensor.sum(dim)
            with self.benchmark(type="sum", dim=dim) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted.sum(dim)
            self._check(encrypted_out, reference, "sum failed")

    def test_div(self):
        """Tests division of encrypted tensor by scalar and tensor."""
        for function in ["div", "div_"]:
            for scalar in [2, 2.0]:
                tensor = get_random_test_tensor(is_float=True)

                reference = tensor.float().div(scalar)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(scalar)
                self._check(encrypted_tensor, reference, "scalar division failed")

                # multiply denominator by 10 to avoid dividing by small num
                divisor = get_random_test_tensor(is_float=True, ex_zero=True) * 10
                reference = tensor.div(divisor)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(divisor)
                self._check(encrypted_tensor, reference, "tensor division failed")

    def test_mean(self):
        """Tests computing means of encrypted tensors."""
        tensor = get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.mean(), tensor.mean(), "mean failed")

        for dim in [0, 1, 2]:
            reference = tensor.mean(dim)
            encrypted_out = encrypted.mean(dim)
            self._check(encrypted_out, reference, "mean failed")

    def test_var(self):
        """Tests computing variances of encrypted tensors."""
        tensor = get_random_test_tensor(size=(5, 10, 15), is_float=True)
        encrypted = MPCTensor(tensor)
        self._check(encrypted.var(), tensor.var(), "var failed")

        for dim in [0, 1, 2]:
            reference = tensor.var(dim)
            encrypted_out = encrypted.var(dim)
            self._check(encrypted_out, reference, "var failed")

    def test_matmul(self):
        """Test matrix multiplication."""
        for tensor_type in [lambda x: x, MPCTensor]:
            tensor = get_random_test_tensor(max_value=7, is_float=True)
            for width in range(2, tensor.nelement()):
                matrix_size = (tensor.nelement(), width)
                matrix = get_random_test_tensor(
                    max_value=7, size=matrix_size, is_float=True
                )
                reference = tensor.matmul(matrix)
                encrypted_tensor = MPCTensor(tensor)
                matrix = tensor_type(matrix)
                encrypted_tensor = encrypted_tensor.matmul(matrix)

                self._check(
                    encrypted_tensor,
                    reference,
                    "Private-%s matrix multiplication failed"
                    % ("private" if tensor_type == MPCTensor else "public"),
                )

    def test_dot_ger(self):
        """Test dot product of vector and encrypted tensor."""
        for tensor_type in [lambda x: x, MPCTensor]:
            tensor1 = get_random_test_tensor(is_float=True).squeeze()
            tensor2 = get_random_test_tensor(is_float=True).squeeze()
            dot_reference = tensor1.dot(tensor2)
            ger_reference = torch.ger(tensor1, tensor2)

            tensor2 = tensor_type(tensor2)

            # dot
            encrypted_tensor = MPCTensor(tensor1)
            encrypted_out = encrypted_tensor.dot(tensor2)
            self._check(
                encrypted_out,
                dot_reference,
                "%s dot product failed" % "private"
                if tensor_type == MPCTensor
                else "public",
            )

            # ger
            encrypted_tensor = MPCTensor(tensor1)
            encrypted_out = encrypted_tensor.ger(tensor2)
            self._check(
                encrypted_out,
                ger_reference,
                "%s outer product failed" % "private"
                if tensor_type == MPCTensor
                else "public",
            )

    def test_squeeze(self):
        tensor = get_random_test_tensor(is_float=True)
        for dim in [0, 1, 2]:
            # Test unsqueeze
            reference = tensor.unsqueeze(dim)

            encrypted = MPCTensor(tensor)
            with self.benchmark(type="unsqueeze", dim=dim) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted.unsqueeze(dim)
            self._check(encrypted_out, reference, "unsqueeze failed")

            # Test squeeze
            encrypted = MPCTensor(tensor.unsqueeze(0))
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
            encrypted_tensor = MPCTensor(tensor)

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
        for func_name in ["conv2d", "conv_transpose2d"]:
            for kernel_type in [lambda x: x, MPCTensor]:
                for matrix_width in range(2, 5):
                    for kernel_width in range(1, matrix_width):
                        for padding in range(kernel_width // 2 + 1):

                            # sample input:
                            matrix_size = (5, matrix_width)
                            matrix = get_random_test_tensor(
                                size=matrix_size, is_float=True
                            )
                            matrix = matrix.unsqueeze(0).unsqueeze(0)

                            # sample filtering kernel:
                            kernel_size = (kernel_width, kernel_width)
                            kernel = get_random_test_tensor(
                                size=kernel_size, is_float=True
                            )
                            kernel = kernel.unsqueeze(0).unsqueeze(0)

                            # perform filtering:
                            encr_matrix = MPCTensor(matrix)
                            encr_kernel = kernel_type(kernel)
                            with self.benchmark(
                                kernel_type=kernel_type.__name__,
                                matrix_width=matrix_width,
                            ) as bench:
                                for _ in bench.iters:
                                    encr_conv = getattr(encr_matrix, func_name)(
                                        encr_kernel, padding=padding
                                    )

                            # check that result is correct:
                            reference = getattr(F, func_name)(
                                matrix, kernel, padding=padding
                            )
                            self._check(encr_conv, reference, "%s failed" % func_name)

    def test_pooling(self):
        """Test avg_pool, sum_pool, max_pool of encrypted tensor."""
        for width in range(2, 5):
            for kernel_size in range(1, width):
                matrix_size = (1, 4, 5, width)
                matrix = get_random_test_tensor(size=matrix_size, is_float=True)
                for stride in range(1, kernel_size + 1):
                    for padding in range(kernel_size // 2 + 1):
                        for func in ["avg_pool2d", "sum_pool2d"]:
                            reference = F.avg_pool2d(
                                matrix, kernel_size, stride=stride, padding=padding
                            )
                            if func == "sum_pool2d":
                                reference *= kernel_size * kernel_size

                            encrypted_matrix = MPCTensor(matrix)
                            with self.benchmark(func=func, width=width) as bench:
                                for _ in bench.iters:
                                    encrypted_pool = getattr(encrypted_matrix, func)(
                                        kernel_size, stride=stride, padding=padding
                                    )
                            self._check(encrypted_pool, reference, "%s failed" % func)

                        # Test max_pool2d
                        for return_indices in [False, True]:
                            kwargs = {
                                "stride": stride,
                                "padding": padding,
                                "return_indices": return_indices,
                            }
                            matrix.requires_grad = True
                            reference = F.max_pool2d(matrix, kernel_size, **kwargs)
                            encrypted_matrix = MPCTensor(matrix)
                            with self.benchmark(
                                func="max_pool2d", width=width
                            ) as bench:
                                for _ in bench.iters:
                                    encrypted_pool = encrypted_matrix.max_pool2d(
                                        kernel_size, **kwargs
                                    )
                            if return_indices:
                                self._check(
                                    encrypted_pool[0], reference[0], "max_pool2d failed"
                                )

                                # Compute max_pool2d backward with random grad
                                grad_output = get_random_test_tensor(
                                    size=reference[0].size(), is_float=True
                                )

                                if matrix.grad is not None:
                                    matrix.grad.data.zero_()
                                reference[0].backward(grad_output)
                                grad_ref = matrix.grad

                                # Compute encrypted backward with same grad
                                encrypted_grad = MPCTensor(grad_output)
                                kwargs = {
                                    "stride": stride,
                                    "padding": padding,
                                    "output_size": encrypted_matrix.size(),
                                }
                                encrypted_grad = encrypted_grad._max_pool2d_backward(
                                    encrypted_pool[1], kernel_size, **kwargs
                                )

                                msg = "max_pool2d_backward failed"
                                self._check(encrypted_grad, grad_ref, msg)
                            else:
                                self._check(
                                    encrypted_pool, reference, "max_pool2d failed"
                                )

    def test_take(self):
        """Tests take function on encrypted tensor"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor([[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long)
        tensor = get_random_test_tensor(size=tensor_size, is_float=True)

        # Test when dimension!=None
        for dimension in range(0, 4):
            reference = torch.from_numpy(tensor.numpy().take(index, dimension))
            encrypted_tensor = MPCTensor(tensor)
            encrypted_out = encrypted_tensor.take(index, dimension)
            self._check(encrypted_out, reference, "take function failed: dimension set")

        # Test when dimension is default (i.e. None)
        sizes = [(15,), (5, 10), (15, 10, 5)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
            take_indices = [[0], [10], [0, 5, 10]]
            for indices in take_indices:
                indices = torch.tensor(indices)
                self._check(
                    encrypted_tensor.take(indices),
                    tensor.take(indices),
                    f"take failed with indices {indices}",
                )

    def test_neg(self):
        """Test negative on encrypted tensor."""
        for width in range(2, 5):
            matrix_size = (5, width)
            matrix = get_random_test_tensor(size=matrix_size, is_float=True)
            encrypted_matrix = MPCTensor(matrix)
            self._check(-encrypted_matrix, -matrix, "__neg__ failed")
            for func_name in ["neg", "neg_"]:
                reference = getattr(matrix, func_name)()
                encrypted_output = getattr(encrypted_matrix, func_name)()
                self._check(encrypted_output, reference, "%s failed" % func_name)

    def test_relu(self):
        """Test relu on encrypted tensor."""
        for width in range(2, 5):
            matrix_size = (5, width)
            matrix = get_random_test_tensor(size=matrix_size, is_float=True)

            # Generate some negative values
            matrix2 = get_random_test_tensor(size=matrix_size, is_float=True)
            matrix = matrix - matrix2

            encrypted_matrix = MPCTensor(matrix)
            reference = F.relu_(matrix)
            with self.benchmark(float=float, width=width, boolean=True) as bench:
                for _ in bench.iters:
                    encrypted_matrix = encrypted_matrix.relu()
            self._check(encrypted_matrix, reference, "relu failed")

    def test_comparators(self):
        """Test comparators (>, >=, <, <=, ==, !=)"""
        for comp in ["gt", "ge", "lt", "le", "eq", "ne"]:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True)

                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor2 = tensor_type(tensor2)

                reference = getattr(tensor, comp)(tensor2).float()

                with self.benchmark(comp=comp) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)(
                            encrypted_tensor2
                        )

                self._check(encrypted_out, reference, "%s comparator failed" % comp)

    def test_max_min(self):
        """Test max and min"""
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        test_cases = [torch.FloatTensor([[1, 1, 2, 1, 4, 1, 3, 4]])] + [
            get_random_test_tensor(size=size, is_float=True) for size in sizes
        ]

        for tensor in test_cases:
            encrypted_tensor = MPCTensor(tensor)
            for comp in ["max", "min"]:
                reference = getattr(tensor, comp)()
                with self.benchmark(niters=10, comp=comp, dim=None) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)()
                self._check(encrypted_out, reference, "%s reduction failed" % comp)

                for dim in range(tensor.dim()):
                    for keepdim in [False, True]:
                        reference = getattr(tensor, comp)(dim=dim, keepdim=keepdim)

                        # Test with one_hot = False
                        with self.benchmark(
                            niters=10,
                            comp=comp,
                            dim=dim,
                            keepdim=keepdim,
                            one_hot=False,
                        ) as bench:
                            for _ in bench.iters:
                                encrypted_out = getattr(encrypted_tensor, comp)(
                                    dim=dim, keepdim=keepdim, one_hot=False
                                )

                        # Check max / min values are correct
                        self._check(
                            encrypted_out[0], reference[0], "%s reduction failed" % comp
                        )

                        # Test argmax / argmin values are correct
                        out_encr = encrypted_out[1]
                        out_decr = out_encr.get_plain_text().long()
                        argmax_ref = reference[1]

                        # Must index into tensor since ties are broken randomly
                        # so crypten and PyTorch can return different indices.
                        # This checks that they index to the same value.
                        if not keepdim:
                            out_decr = out_decr.unsqueeze(dim)
                            argmax_ref = argmax_ref.unsqueeze(dim)
                        mpc_result = tensor.gather(dim, out_decr)
                        torch_result = tensor.gather(dim, argmax_ref)
                        self.assertTrue(
                            (mpc_result == torch_result).all().item(),
                            "%s reduction failed" % comp,
                        )

                        # Test indices with one_hot = True
                        with self.benchmark(
                            niters=10, comp=comp, dim=dim, keepdim=keepdim, one_hot=True
                        ) as bench:
                            for _ in bench.iters:
                                encrypted_out = getattr(encrypted_tensor, comp)(
                                    dim=dim, keepdim=keepdim, one_hot=True
                                )

                        # Check argmax results
                        val_ref = reference[0]
                        out_encr = encrypted_out[1]
                        out_decr = out_encr.get_plain_text()
                        self.assertTrue((out_decr.sum(dim=dim) == 1).all())
                        self.assertTrue(
                            (
                                out_decr.mul(tensor).sum(dim=dim, keepdim=keepdim)
                                == val_ref
                            ).all()
                        )

    def test_argmax_argmin(self):
        """Test argmax and argmin"""
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        test_cases = [torch.FloatTensor([[1, 1, 2, 1, 4, 1, 3, 4]])] + [
            get_random_test_tensor(size=size, is_float=True) for size in sizes
        ]

        for tensor in test_cases:
            encrypted_tensor = MPCTensor(tensor)
            for comp in ["argmax", "argmin"]:
                cmp = comp[3:]

                value = getattr(tensor, cmp)()

                # test with one_hot = False
                with self.benchmark(
                    niters=10, comp=comp, dim=None, one_hot=False
                ) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)(one_hot=False)

                # Must index into tensor since ties are broken randomly
                # so crypten and PyTorch can return different indices.
                # This checks that they index to the same value.
                decrypted_out = encrypted_out.get_plain_text()
                if tensor.dim() == 0:  # if input is 0-d, argmax should be 0
                    self.assertEqual(decrypted_out, 0)
                else:
                    decrypted_val = tensor.flatten()[decrypted_out.long()]
                    self.assertTrue(decrypted_val.eq(value).all().item())

                # test with one_hot = False
                with self.benchmark(
                    niters=10, comp=comp, dim=None, one_hot=True
                ) as bench:
                    for _ in bench.iters:
                        encrypted_out = getattr(encrypted_tensor, comp)(one_hot=True)
                one_hot_indices = (tensor == value).float()
                decrypted_out = encrypted_out.get_plain_text()
                self.assertTrue(decrypted_out.sum() == 1)
                self.assertTrue(decrypted_out.mul(one_hot_indices).sum() == 1)

                for dim in range(tensor.dim()):
                    for keepdim in [False, True]:
                        # Compute one-hot argmax/min reference in plaintext
                        values, indices = getattr(tensor, cmp)(dim=dim, keepdim=keepdim)

                        # test with one_hot = False
                        with self.benchmark(
                            niters=10, comp=comp, dim=dim, one_hot=False
                        ) as bench:
                            for _ in bench.iters:
                                encrypted_out = getattr(encrypted_tensor, comp)(
                                    dim=dim, keepdim=keepdim, one_hot=False
                                )

                        # Must index into tensor since ties are broken randomly
                        # so crypten and PyTorch can return different indices.
                        # This checks that they index to the same value.abs
                        decrypted_out = encrypted_out.get_plain_text()
                        if not keepdim:
                            decrypted_out = decrypted_out.unsqueeze(dim)
                            indices = indices.unsqueeze(dim)
                        decrypted_val = tensor.gather(dim, decrypted_out.long())
                        reference = tensor.gather(dim, indices)
                        self.assertTrue(decrypted_val.eq(reference).all().item())

                        # test with one_hot = True
                        with self.benchmark(
                            niters=10, comp=comp, dim=dim, one_hot=True
                        ) as bench:
                            for _ in bench.iters:
                                encrypted_out = getattr(encrypted_tensor, comp)(
                                    dim=dim, keepdim=keepdim, one_hot=True
                                )
                        decrypted_out = encrypted_out.get_plain_text()

                        if not keepdim:
                            values = values.unsqueeze(dim)
                        one_hot_indices = tensor.eq(values).float()
                        self.assertTrue(decrypted_out.sum(dim=dim).eq(1).all())
                        self.assertTrue(
                            decrypted_out.mul(one_hot_indices).sum(dim=dim).eq(1).all()
                        )

    def test_abs_sign(self):
        """Test absolute value function"""
        for op in ["abs", "sign"]:
            tensor = get_random_test_tensor(is_float=True)
            if op == "sign":
                # do not test on 0 since torch.tensor([0]).sign() = 0
                tensor = tensor + (tensor == 0).float()
            encrypted_tensor = MPCTensor(tensor)
            reference = getattr(tensor, op)()

            with self.benchmark(niters=10, op=op) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, op)()

            self._check(encrypted_out, reference, "%s failed" % op)

    def test_approximations(self):
        """Test appoximate functions (exp, log, sqrt, reciprocal, pos_pow)"""

        def test_with_inputs(func, input):
            encrypted_tensor = MPCTensor(input)
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
        encrypted_tensor = MPCTensor(tensor)

        # Reduced the max_value so approximations have less absolute error
        tensor_exponent = get_random_test_tensor(
            max_value=2, size=tensor.size(), is_float=True
        )
        exponents = [-3, -2, -1, 0, 1, 2, 3, tensor_exponent]
        exponents += [MPCTensor(tensor_exponent)]
        for p in exponents:
            if isinstance(p, MPCTensor):
                reference = tensor.pow(p.get_plain_text())
            else:
                reference = tensor.pow(p)
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor.pos_pow(p)
            self._check(encrypted_out, reference, f"pos_pow failed with power {p}")

    def test_pow(self):
        """Tests pow function"""
        tensor = get_random_test_tensor(is_float=True)
        encrypted_tensor = MPCTensor(tensor)

        for power in [-3, -2, -1, 0, 1, 2, 3]:
            reference = tensor.pow(power)
            with self.benchmark(niters=10, func="pow", power=power) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor.pow(power)
            self._check(encrypted_out, reference, "pow failed with power %s" % power)

    def test_norm(self):
        """Tests p-norm"""
        for p in [1, 1.5, 2, 3, float("inf"), "fro"]:
            for dim in [None, 0, 1, 2]:
                tensor = get_random_test_tensor(size=(3, 3, 3), is_float=True) / 5
                if dim is None:
                    reference = tensor.norm(p=p)
                else:
                    reference = tensor.norm(p=p, dim=dim)

                encrypted = MPCTensor(tensor)
                with self.benchmark() as bench:
                    for _ in bench.iters:
                        encrypted_out = encrypted.norm(p=p, dim=dim)
                self._check(encrypted_out, reference, f"{p}-norm failed", tolerance=0.5)

    def test_logistic(self):
        """Tests logistic functions (sigmoid, tanh)"""
        tensor = torch.tensor([0.01 * i for i in range(-1000, 1001, 1)])
        encrypted_tensor = MPCTensor(tensor)

        cases = ["sigmoid", "tanh"]
        for func in cases:
            reference = getattr(tensor, func)()
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_cos_sin(self):
        """Tests trigonometric functions (cos, sin)"""
        tensor = torch.tensor([0.01 * i for i in range(-1000, 1001, 1)])
        encrypted_tensor = MPCTensor(tensor)

        cases = ["cos", "sin"]
        for func in cases:
            reference = getattr(tensor, func)()
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_bernoulli(self):
        """Tests bernoulli sampling"""
        for size in [(10,), (10, 10), (10, 10, 10)]:
            probs = MPCTensor(torch.rand(size))
            with self.benchmark(size=size) as bench:
                for _ in bench.iters:
                    randvec = probs.bernoulli()
            self.assertTrue(randvec.size() == size, "Incorrect size")
            tensor = randvec.get_plain_text()
            self.assertTrue(((tensor == 0) + (tensor == 1)).all(), "Invalid values")

        probs = MPCTensor(torch.Tensor(int(1e6)).fill_(0.2))
        randvec = probs.bernoulli().get_plain_text()
        frac_zero = float((randvec == 0).sum()) / randvec.nelement()
        self.assertTrue(math.isclose(frac_zero, 0.8, rel_tol=1e-3, abs_tol=1e-3))

    def test_softmax(self):
        """Test softmax function"""
        # Test 0-dim tensor:
        tensor = get_random_test_tensor(size=(), is_float=True)
        reference = tensor.softmax(0)
        encrypted_tensor = MPCTensor(tensor)
        with self.benchmark(size=(), dim=0) as bench:
            for _ in bench.iters:
                encrypted_out = encrypted_tensor.softmax(0)
        self._check(encrypted_out, reference, "softmax failed")

        # Test all other sizes
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
            (1, 5, 5, 5),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True) / 5
            encrypted_tensor = MPCTensor(tensor)

            for dim in range(tensor.dim()):
                reference = tensor.softmax(dim)
                with self.benchmark(size=size, dim=dim) as bench:
                    for _ in bench.iters:
                        encrypted_out = encrypted_tensor.softmax(dim)

                self._check(encrypted_out, reference, "softmax failed")

    def test_get_set(self):
        """Tests element setting and getting by index"""
        for tensor_type in [lambda x: x, MPCTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = get_random_test_tensor(size=(size, size), is_float=True)
                reference = tensor[:, 0]

                encrypted_tensor = MPCTensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,), is_float=True)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = MPCTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = MPCTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

    def test_pad(self):
        """Tests padding"""
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
            encrypted_tensor = MPCTensor(tensor)

            for pad in pads:
                for value in [0, 1, 10]:
                    for tensor_type in [lambda x: x, MPCTensor]:
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
                    encrypted = MPCTensor(tensor1)
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
                        encrypted = MPCTensor(tensor1)
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

    def test_broadcast_arithmetic_ops(self):
        """Test broadcast of arithmetic functions."""
        arithmetic_functions = ["add", "sub", "mul", "div"]
        # TODO: Add broadcasting for pos_pow since it can take a tensor argument
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

        for tensor_type in [lambda x: x, MPCTensor]:
            for func in arithmetic_functions:
                for size1, size2 in itertools.combinations(arithmetic_sizes, 2):
                    exclude_zero = True if func == "div" else False
                    # multiply denominator by 10 to avoid dividing by small num
                    const = 10 if func == "div" else 1

                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = get_random_test_tensor(
                        size=size2, is_float=True, ex_zero=exclude_zero
                    )
                    tensor2 *= const
                    encrypted1 = MPCTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(tensor2)
                    encrypted_out = getattr(encrypted1, func)(encrypted2)

                    private = isinstance(encrypted2, MPCTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s %s broadcast failed"
                        % ("private" if private else "public", func),
                    )

                    # Test with integer tensor
                    tensor2 = get_random_test_tensor(
                        size=size2, is_float=False, ex_zero=exclude_zero
                    )
                    tensor2 *= const
                    reference = getattr(tensor1, func)(tensor2.float())
                    encrypted_out = getattr(encrypted1, func)(tensor2)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s broadcast failed with public integer tensor" % func,
                    )

    def test_broadcast_matmul(self):
        """Test broadcast of matmul."""
        matmul_sizes = [(1, 1), (1, 5), (5, 1), (5, 5)]
        batch_dims = [(), (1,), (5,), (1, 1), (1, 5), (5, 5)]

        for tensor_type in [lambda x: x, MPCTensor]:

            for size in matmul_sizes:
                for batch1, batch2 in itertools.combinations(batch_dims, 2):
                    size1 = (*batch1, *size)
                    size2 = (*batch2, *size)

                    tensor1 = get_random_test_tensor(size=size1, is_float=True)
                    tensor2 = get_random_test_tensor(size=size2, is_float=True)
                    tensor2 = tensor2.transpose(-2, -1)

                    encrypted1 = MPCTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)

                    reference = tensor1.matmul(tensor2)
                    encrypted_out = encrypted1.matmul(encrypted2)
                    private = isinstance(encrypted2, MPCTensor)
                    self._check(
                        encrypted_out,
                        reference,
                        "%s matmul broadcast failed"
                        % ("private" if private else "public"),
                    )

                    # Test with integer tensor
                    tensor2 = get_random_test_tensor(size=size2, is_float=False)
                    tensor2 = tensor2.float().transpose(-2, -1)
                    reference = tensor1.matmul(tensor2)
                    encrypted_out = encrypted1.matmul(tensor2)
                    self._check(
                        encrypted_out,
                        reference,
                        "matmul broadcast failed with public integer tensor",
                    )

    def test_inplace(self):
        """Test inplace vs. out-of-place functions"""
        for op in ["add", "sub", "mul", "div"]:
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = get_random_test_tensor(is_float=True)
                tensor2 = get_random_test_tensor(is_float=True)

                reference = getattr(torch, op)(tensor1, tensor2)

                encrypted1 = MPCTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                input_tensor_id = id(encrypted1._tensor)
                input_encrypted_id = id(encrypted1)

                # Test that out-of-place functions do not modify the input
                private = isinstance(encrypted2, MPCTensor)
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
                self.assertFalse(id(encrypted_out._tensor) == input_tensor_id)
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
                self.assertTrue(id(encrypted_out._tensor) == input_tensor_id)
                self.assertTrue(id(encrypted_out) == input_encrypted_id)

    def test_copy_clone(self):
        """Tests shallow_copy and clone of encrypted tensors."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            # test shallow_copy
            encrypted_tensor_shallow = encrypted_tensor.shallow_copy()
            self.assertEqual(
                id(encrypted_tensor_shallow._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_shallow, tensor, "shallow_copy failed")
            # test clone
            encrypted_tensor_clone = encrypted_tensor.clone()
            self.assertNotEqual(
                id(encrypted_tensor_clone._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_clone, tensor, "clone failed")

    def test_index_select(self):
        """Tests index_select of encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)
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
            encr_tensor = MPCTensor(tensor)
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
            encrypted_tensor = MPCTensor(tensor)

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
            encrypted_tensor = MPCTensor(tensor)
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
            encrypted_tensor = MPCTensor(tensor)
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
            encrypted_tensor = MPCTensor(tensor)

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
        encrypted_tensor = MPCTensor(tensor)
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
            encrypted_tensor = MPCTensor(tensor)

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
            encrypted_tensor = MPCTensor(tensor)

            self._check(encrypted_tensor.trace(), tensor.trace(), "trace failed")

    def test_flip(self):
        """Tests flip operation on encrypted tensors."""
        sizes = [(5,), (5, 10), (5, 10, 15)]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

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
        encrypted_tensor = MPCTensor(tensor)
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
            encrypted_tensor1 = MPCTensor(tensor1)
            tensor2 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                get_random_test_tensor(max_value=1, size=size, is_float=False) + 1
            )
            condition_encrypted = MPCTensor(condition_tensor)
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
            encrypted = MPCTensor(tensor)

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

    def test_unbind(self):
        """Tests unbind"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted = MPCTensor(tensor)
            for dim in range(tensor.dim()):
                print(dim)
                reference = tensor.unbind(dim)
                encrypted_out = encrypted.unbind(dim)

                self._check_tuple(encrypted_out, reference, "unbind failed")

    def test_split(self):
        """Tests split"""
        sizes = [
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 5),
            (1, 1, 1),
            (5, 5, 5),
            (1, 1, 1, 1),
            (5, 5, 5, 5),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=True)
            encrypted = MPCTensor(tensor)
            for dim in range(tensor.dim()):
                # Get random split
                split = get_random_test_tensor(size=(), max_value=tensor.size(dim))
                split = split.abs().clamp(0, tensor.size(dim) - 1)
                split = split.item()

                # Test int split
                int_split = 1 if split == 0 else split
                reference = tensor.split(int_split, dim=dim)
                encrypted_out = encrypted.split(int_split, dim=dim)
                self._check_tuple(encrypted_out, reference, "split failed")

                # Test list split
                split = [split, tensor.size(dim) - split]
                reference = tensor.split(split, dim=dim)
                encrypted_out = encrypted.split(split, dim=dim)
                self._check_tuple(encrypted_out, reference, "split failed")

    # TODO: Write the following unit tests
    @unittest.skip("Test not implemented")
    def test_gather_scatter(self):
        pass


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestMPC.benchmarks_enabled = True
    unittest.main()
