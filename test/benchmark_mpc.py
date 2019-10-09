#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import torch
from crypten.common.tensor_types import is_float_tensor, is_int_tensor
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor


class MPCBenchmark(MultiProcessTestCase):

    benchmarks_enabled = True

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        crypten.init()

        torch.manual_seed(0)

        self.sizes = [(1, 8), (1, 16), (1, 32)]

        self.int_tensors = [
            get_random_test_tensor(size=size, is_float=False) for size in self.sizes
        ]
        self.int_operands = [
            (
                get_random_test_tensor(size=size, is_float=False),
                get_random_test_tensor(size=size, is_float=False),
            )
            for size in self.sizes
        ]
        self.float_tensors = [
            get_random_test_tensor(size=size, is_float=True) for size in self.sizes
        ]
        self.float_operands = [
            (
                get_random_test_tensor(size=size, is_float=True),
                get_random_test_tensor(size=size, is_float=True),
            )
            for size in self.sizes
        ]

        self.tensors = self.int_tensors + self.float_tensors
        self.operands = self.int_operands + self.float_operands
        self.sizes = self.sizes + self.sizes

    @staticmethod
    def is_float(tensor):
        return tensor.dtype in [torch.float64, torch.float32, torch.float16]

    def test_encrypt(self):
        for size, tensor in zip(self.sizes, self.tensors):
            if is_float_tensor(tensor):
                tensor_type = ArithmeticSharedTensor
            else:
                tensor_type = BinarySharedTensor
            tensor_name = tensor_type.__name__
            with self.benchmark(
                tensor_type=tensor_name, size=size, is_float=self.is_float(tensor)
            ) as bench:
                for _ in bench.iters:
                    encrypted_tensor = tensor_type(tensor)

            self.assertTrue(encrypted_tensor is not None)

    def test_decrypt(self):
        for tensor_type in [ArithmeticSharedTensor, BinarySharedTensor]:
            tensor_name = tensor_type.__name__
            tensors = self.tensors
            if tensor_type == ArithmeticSharedTensor:
                tensors = [t for t in tensors if is_float_tensor(t)]
            else:
                tensors = [t for t in tensors if is_int_tensor(t)]
            encrypted_tensors = [tensor_type(tensor) for tensor in tensors]
            data = zip(self.sizes, tensors, encrypted_tensors)
            for size, tensor, encrypted_tensor in data:
                with self.benchmark(
                    tensor_type=tensor_name, size=size, float=self.is_float(tensor)
                ) as bench:
                    for _ in bench.iters:
                        tensor = encrypted_tensor.get_plain_text()

    def test_arithmetic(self):
        arithmetic_functions = ["add", "add_", "sub", "sub_", "mul", "mul_"]
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, ArithmeticSharedTensor]:
                for size, (a, b) in zip(self.sizes, self.float_operands):
                    encrypted_as = [
                        ArithmeticSharedTensor(a) for _ in range(self.benchmark_iters)
                    ]
                    encrypted_bs = [tensor_type(b) for _ in range(self.benchmark_iters)]

                    data = list(zip(encrypted_as, encrypted_bs))
                    with self.benchmark(
                        data=data, func=func, float=True, size=size
                    ) as bench:
                        for encrypted_a, encrypted_b in bench.data:
                            encrypted_out = getattr(encrypted_a, func)(encrypted_b)

                    self.assertTrue(encrypted_out is not None)

    def test_div_scalar(self):
        for function in ["div", "div_"]:
            for size, tensor in zip(self.sizes, self.float_tensors):
                scalar = 2.0 if self.is_float(tensor) else 2

                # Copy the tensors for each benchmark iteration because div_
                # mutates the data
                encrypted_tensors = [
                    ArithmeticSharedTensor(tensor) for _ in range(self.benchmark_iters)
                ]
                with self.benchmark(
                    data=encrypted_tensors, func=function, float=True, size=size
                ) as bench:
                    for encrypted_tensor in bench.data:
                        result = getattr(encrypted_tensor, function)(scalar)

                self.assertTrue(result is not None)

    def test_matmul(self):
        def do_benchmark():
            matrix_size = (_tensor.nelement(), _width)
            matrix = get_random_test_tensor(
                max_value=7, size=matrix_size, is_float=True
            )
            niters = 10
            encrypted_tensors = [ArithmeticSharedTensor(_tensor) for _ in range(niters)]
            matrix = _tensor_type(matrix)

            with self.benchmark(
                data=encrypted_tensors,
                function=_function,
                tensor_type=_tensor_type.__name__,
                n=_size[0],
                m=_size[1],
                k=_width,
            ) as bench:
                for encrypted_tensor in bench.data:
                    encrypted_tensor = getattr(encrypted_tensor, _function)(matrix)

            self.assertTrue(encrypted_tensor is not None)

        _function = "matmul"
        for _tensor_type in [lambda x: x, ArithmeticSharedTensor]:
            for _size, _tensor in zip(self.sizes, self.float_tensors):
                for _width in [4, 8]:
                    do_benchmark()
