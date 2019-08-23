#!/usr/bin/env python3

import itertools

# dependencies:
import logging
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import torch
import torch.nn.functional as F


# placeholders for class references, to be filled later by import_crypten():
ArithmeticSharedTensor, is_float_tensor = None, None


def import_crypten():
    """
    Imports CrypTen types. This function is called after environment variables
    in MultiProcessTestCase.setUp() are set, and sets the class references for
    all test functions.
    """
    global ArithmeticSharedTensor, is_float_tensor
    from crypten.primitives.arithmetic.arithmetic import (
        ArithmeticSharedTensor as _ArithmeticSharedTensor,
    )
    from crypten.common.tensor_types import is_float_tensor as _is_float_tensor

    ArithmeticSharedTensor = _ArithmeticSharedTensor
    is_float_tensor = _is_float_tensor


class TestArithmetic(MultiProcessTestCase):
    """
        This class tests all functions of the ArithmeticSharedTensor.
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
        if self.rank >= 0:
            import_crypten()

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
        from crypten.common.util import count_wraps
        from crypten.common.sharing import share
        from crypten import comm

        num_parties = int(self.world_size)
        tensor = get_random_test_tensor(is_float=False)
        shares = share(tensor, num_parties=num_parties)

        # Note: This test relies on count_wraps function being correct
        reference = count_wraps(shares)

        # Sync shares between parties
        share = comm.scatter(shares, 0)

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

    def test_approximations(self):
        """Test appoximate functions (exp, log, sqrt, reciprocal, pow)"""
        tensor = torch.tensor([0.01 * i for i in range(1, 1001, 1)])
        encrypted_tensor = ArithmeticSharedTensor(tensor)

        cases = ["exp", "log", "sqrt", "reciprocal"]
        for func in cases:
            reference = getattr(tensor, func)()
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

        for power in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            reference = tensor.pow(power)
            with self.benchmark(niters=10, func="pow", power=power) as bench:
                for _ in bench.iters:
                    encrypted_out = encrypted_tensor.pow(power)
            self._check(encrypted_out, reference, "pow failed with %s power" % power)

    def test_norm(self):
        # Test 2-norm
        tensor = get_random_test_tensor(is_float=True)
        reference = tensor.norm()

        encrypted = ArithmeticSharedTensor(tensor)
        with self.benchmark() as bench:
            for _ in bench.iters:
                encrypted_out = encrypted.norm()
        self._check(encrypted_out, reference, "2-norm failed", tolerance=0.5)

    def test_cos_sin(self):
        tensor = torch.tensor([0.01 * i for i in range(-800, 801, 1)])
        encrypted_tensor = ArithmeticSharedTensor(tensor)

        cases = ["cos", "sin"]
        for func in cases:
            reference = getattr(tensor, func)()
            with self.benchmark(niters=10, func=func) as bench:
                for _ in bench.iters:
                    encrypted_out = getattr(encrypted_tensor, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_softmax(self):
        """Test max function"""
        tensor = get_random_test_tensor(is_float=True)
        reference = torch.nn.functional.softmax(tensor, dim=1)

        encrypted_tensor = ArithmeticSharedTensor(tensor)
        with self.benchmark() as bench:
            for _ in bench.iters:
                encrypted_out = encrypted_tensor.softmax()
        self._check(encrypted_out, reference, "softmax failed")

    def test_onnx_gather(self):
        """Tests onnx_gather function of encrypted tensor"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor([[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long)
        tensor = get_random_test_tensor(size=tensor_size, is_float=True)
        for dimension in range(0, 4):
            reference = torch.from_numpy(tensor.numpy().take(index, dimension))
            encrypted_tensor = ArithmeticSharedTensor(tensor)
            encrypted_out = encrypted_tensor.onnx_gather(index, dimension)
            self._check(encrypted_out, reference, "onnx_gather function failed")

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
                    if func == "div" and tensor_type == ArithmeticSharedTensor:
                        tensor2 = tensor2.abs()

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
                    tensor2 = tensor2.abs()

                reference = getattr(torch, op)(tensor1, tensor2)

                encrypted1 = ArithmeticSharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                input_plain_id = id(encrypted1._tensor)
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
                self.assertFalse(id(encrypted_out._tensor) == input_plain_id)
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
                self.assertTrue(id(encrypted_out._tensor) == input_plain_id)
                self.assertTrue(id(encrypted_out) == input_encrypted_id)


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestArithmetic.benchmarks_enabled = True
    unittest.main()
