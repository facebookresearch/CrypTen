#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import collections
import itertools
import logging
import unittest

import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F
from crypten.config import cfg
from crypten.cuda import CUDALongTensor
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase
from test.test_mpc import TestMPC


class MLP(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x


@unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
class TestCUDA(TestMPC):
    """
    This class tests all functions of CUDALongTensor as well as its integration with MPCTensor.
    """

    def _check_int(self, result, reference, msg):
        # Check sizes match
        self.assertTrue(result.size() == reference.size(), msg)

        is_eq = (result == reference).all().item() == 1

        if not is_eq:
            logging.info(msg)
            logging.info("Result %s" % result)
            logging.info("Reference %s" % reference)
            logging.info("Result - Reference = %s" % (result - reference))

        self.assertTrue(is_eq, msg=msg)

    def test_mlp(self) -> None:
        """Test the forward/backward pass of MLP on GPU"""
        model = MLP()
        dummy_input = torch.empty((32, 128))
        model = crypten.nn.from_pytorch(model, dummy_input=dummy_input)
        model = model.to(self.device)
        model.encrypt()
        model.train()

        rand_in = crypten.cryptensor(
            torch.rand([32, 128], device=self.device), requires_grad=True
        )
        output = model(rand_in)

        model.zero_grad()
        output.backward()
        model.update_parameters(learning_rate=1e-3)

    def test_patched_matmul(self) -> None:
        """Test torch.matmul on CUDALongTensor"""
        input_sizes = [
            (5,),
            (5, 5),
            (5,),
            (5, 5),
            (5, 5, 5),
            (5,),
            (5, 5, 5, 5),
            (5, 5),
            # Check large interleaves for 4x16-bit process
            (1, 256),
            (1, 1024),
        ]
        other_sizes = [
            (5,),
            (5, 5),
            (5, 5),
            (5,),
            (5,),
            (5, 5, 5),
            (5, 5),
            (5, 5, 5, 5),
            (256, 1),
            (1024, 1),
        ]

        for x_size, y_size in zip(input_sizes, other_sizes):
            x = get_random_test_tensor(size=x_size, max_value=2**62, is_float=False)
            x_cuda = CUDALongTensor(x)

            y = get_random_test_tensor(size=y_size, max_value=2**62, is_float=False)
            y_cuda = CUDALongTensor(y)

            z = torch.matmul(x_cuda, y_cuda)
            self.assertTrue(
                type(z) == CUDALongTensor, "result should be a CUDALongTensor"
            )

            reference = torch.matmul(x, y)
            self._check_int(z.cpu(), reference, "matmul failed for cuda_patches")

    def test_conv1d_smaller_signal_one_channel(self) -> None:
        self._patched_conv1d(5, 1)
        self._conv1d(5, 1)

    def test_conv1d_smaller_signal_many_channels(self) -> None:
        self._patched_conv1d(5, 5)
        self._conv1d(5, 5)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_larger_signal_one_channel(self) -> None:
        self._patched_conv1d(16, 1)
        self._conv1d(16, 1)

    def test_conv1d_larger_signal_many_channels(self) -> None:
        self._patched_conv1d(16, 5)
        self._conv1d(16, 5)

    def test_conv1d_large_filter(self) -> None:
        self._patched_conv1d(1024, 1, kernel_sizes=[256, 512])

    def _patched_conv1d(self, signal_size, in_channels, kernel_sizes=None):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
        nbatches = [1, 3]
        ochannels = [1, 3, 6]
        paddings = [0, 1]
        strides = [1, 2]

        if kernel_sizes is None:
            kernel_sizes = [1, 2, 3]

        for func_name in ["conv1d", "conv_transpose1d"]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
            ) in itertools.product(
                nbatches, kernel_sizes, ochannels, paddings, strides
            ):
                input_size = (batches, in_channels, signal_size)
                signal = get_random_test_tensor(size=input_size, is_float=False)

                if func_name == "conv1d":
                    k_size = (out_channels, in_channels, kernel_size)
                else:
                    k_size = (in_channels, out_channels, kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                signal_cuda = CUDALongTensor(signal)
                kernel_cuda = CUDALongTensor(kernel)

                reference = getattr(F, func_name)(
                    signal, kernel, padding=padding, stride=stride
                )
                result = getattr(F, func_name)(
                    signal_cuda, kernel_cuda, padding=padding, stride=stride
                )
                self.assertTrue(
                    type(result) == CUDALongTensor, "result should be a CUDALongTensor"
                )
                result = result.data.cpu()

                self._check_int(result, reference, f"{func_name} failed")

    def test_conv2d_square_image_one_channel(self) -> None:
        self._patched_conv2d((5, 5), 1)
        self._conv2d((5, 5), 1)

    def test_conv2d_square_image_many_channels(self) -> None:
        self._patched_conv2d((5, 5), 5)
        self._conv2d((5, 5), 5)

    def test_conv2d_rectangular_image_one_channel(self) -> None:
        self._patched_conv2d((16, 7), 1)
        self._conv2d((16, 7), 1)

    def test_conv2d_rectangular_image_many_channels(self) -> None:
        self._patched_conv2d((16, 7), 5)
        self._conv2d((16, 7), 5)

    def test_conv2d_large_kernel(self) -> None:
        self.nbatches = [1]
        self.ochannels = [1]
        self.paddings = [0]
        self.strides = [(64, 64)]
        self.kernel_sizes = [(64, 64)]
        self._patched_conv2d((64, 64), 1)

    def _patched_conv2d(self, image_size, in_channels):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
        kwargs = collections.OrderedDict()
        kwargs["nbatches"] = [1, 3]
        kwargs["kernel_sizes"] = [(1, 1), (2, 2), (2, 3)]
        kwargs["ochannels"] = [1, 3, 6]
        kwargs["paddings"] = [0, 1, (0, 1)]
        kwargs["strides"] = [1, 2, (1, 2)]

        for attribute in [
            "nbatches",
            "ochannels",
            "paddings",
            "strides",
            "kernel_sizes",
        ]:
            if hasattr(self, attribute):
                kwargs[attribute] = getattr(self, attribute)

        for func_name in ["conv2d", "conv_transpose2d"]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
            ) in itertools.product(*[v for _, v in kwargs.items()]):

                # sample input:
                input_size = (batches, in_channels, *image_size)
                input = get_random_test_tensor(size=input_size, is_float=False)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels, in_channels, *kernel_size)
                else:
                    k_size = (in_channels, out_channels, *kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                input_cuda = CUDALongTensor(input)
                kernel_cuda = CUDALongTensor(kernel)

                result = getattr(F, func_name)(
                    input_cuda, kernel_cuda, padding=padding, stride=stride
                )
                self.assertTrue(
                    type(result) == CUDALongTensor, "result should be a CUDALongTensor"
                )
                result = result.data.cpu()

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input, kernel, padding=padding, stride=stride
                )
                self._check_int(result, reference, "%s failed" % func_name)

    def test_torch_arithmetic(self) -> None:
        """Test torch arithmetic on CUDALongTensor"""
        funcs = ["add", "sub", "mul", "div"]
        a = get_random_test_tensor(is_float=False)
        b = get_random_test_tensor(min_value=1, is_float=False)

        a_cuda = CUDALongTensor(a)
        b_cuda = CUDALongTensor(b)

        for op in funcs:
            kwargs = {"rounding_mode": "trunc"} if op == "div" else {}

            reference = getattr(torch, op)(a, b, **kwargs)
            result = getattr(torch, op)(a_cuda, b_cuda, **kwargs)
            result2 = getattr(a_cuda, op)(b_cuda, **kwargs)

            self.assertTrue(type(result), CUDALongTensor)
            self._check_int(
                reference, result.cpu(), "torch.{} failed for CUDALongTensor".format(op)
            )
            self._check_int(
                reference,
                result2.cpu(),
                "torch.{} failed for CUDALongTensor".format(op),
            )

    def test_torch_comparators(self) -> None:
        """Test torch comparators on CUDALongTensor"""
        for comp in ["gt", "ge", "lt", "le", "eq", "ne"]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)

            t_cuda = CUDALongTensor(tensor)
            t2_cuda = CUDALongTensor(tensor2)

            reference = getattr(torch, comp)(tensor, tensor2).long()
            result1 = getattr(t_cuda, comp)(t2_cuda)
            result2 = getattr(torch, comp)(t_cuda, t2_cuda)

            self.assertTrue(
                type(result1) == CUDALongTensor, "result should be a CUDALongTensor"
            )
            self.assertTrue(
                type(result2) == CUDALongTensor, "result should be a CUDALongTensor"
            )
            self._check_int(result1.cpu(), reference, "%s comparator failed" % comp)
            self._check_int(result2.cpu(), reference, "%s comparator failed" % comp)

    def test_torch_avg_pool2d(self) -> None:
        """Test avg_pool2d on CUDALongTensor"""
        for width in range(2, 5):
            for kernel_size in range(1, width):
                matrix_size = (1, 4, 5, width)
                matrix = get_random_test_tensor(size=matrix_size, is_float=False)
                matrix_cuda = CUDALongTensor(matrix)
                for stride in range(1, kernel_size + 1):
                    for padding in range(kernel_size // 2 + 1):
                        for divisor_override in [None, 1, 2]:
                            reference = F.avg_pool2d(
                                matrix,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                divisor_override=divisor_override,
                            )
                            result = F.avg_pool2d(
                                matrix_cuda,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                divisor_override=divisor_override,
                            )

                            self._check_int(
                                result.cpu(), reference, "avg_pool2d failed"
                            )

    def test_torch_stack_cat(self) -> None:
        """Test torch.cat/torch.stack on CUDALongTensor"""
        funcs = ["stack", "cat"]

        tensors = [get_random_test_tensor(is_float=False) for _ in range(10)]
        tensors_cuda = [CUDALongTensor(t) for t in tensors]

        for op in funcs:
            reference = getattr(torch, op)(tensors)
            result = getattr(CUDALongTensor, op)(tensors_cuda)

            self.assertTrue(
                type(result) == CUDALongTensor, "result should be a CUDALongTensor"
            )
            self._check_int(
                reference, result.cpu(), "torch.{} failed for CUDALongTensor".format(op)
            )

    def test_torch_broadcast_tensor(self) -> None:
        """Test torch.broadcast_tensor on CUDALongTensor"""
        x = get_random_test_tensor(size=(1, 5), is_float=False)
        y = get_random_test_tensor(size=(5, 1), is_float=False)

        x_cuda = CUDALongTensor(x)
        y_cuda = CUDALongTensor(y)

        a, b = torch.broadcast_tensors(x, y)
        a_cuda, b_cuda = torch.broadcast_tensors(x_cuda, y_cuda)

        self.assertTrue(
            type(a_cuda) == CUDALongTensor, "result should be a CUDALongTensor"
        )
        self.assertTrue(
            type(b_cuda) == CUDALongTensor, "result should be a CUDALongTensor"
        )
        self._check_int(
            a, a_cuda.cpu(), "torch.broadcast_tensor failed for CUDALongTensor"
        )
        self._check_int(
            b, b_cuda.cpu(), "torch.broadcast_tensor failed for CUDALongTensor"
        )

    def test_torch_split(self) -> None:
        """Test torch.split on CUDALongTensor"""
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
            tensor = get_random_test_tensor(size=size, is_float=False)
            t_cuda = CUDALongTensor(tensor)
            for dim in range(tensor.dim()):
                # Get random split
                split = get_random_test_tensor(size=(), max_value=tensor.size(dim))
                split = split.abs().clamp(0, tensor.size(dim) - 1)
                split = split.item()

                # Test int split
                int_split = 1 if split == 0 else split
                reference = torch.split(tensor, int_split, dim=dim)
                result = t_cuda.split(int_split, dim=dim)
                result2 = torch.split(t_cuda, int_split, dim=dim)

                for i in range(len(result)):
                    self.assertTrue(
                        type(result[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self.assertTrue(
                        type(result2[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self._check_int(result[i].cpu(), reference[i], "split failed")
                    self._check_int(result2[i].cpu(), reference[i], "split failed")

                # Test list split
                split = [split, tensor.size(dim) - split]
                reference = torch.split(tensor, split, dim=dim)
                result = t_cuda.split(split, dim=dim)
                result2 = torch.split(t_cuda, split, dim=dim)

                for i in range(len(result)):
                    self.assertTrue(
                        type(result[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self.assertTrue(
                        type(result2[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self._check_int(result[i].cpu(), reference[i], "split failed")
                    self._check_int(result2[i].cpu(), reference[i], "split failed")

    def test_torch_unbind(self) -> None:
        """Test torch.unbind on CUDALongTensor"""
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
            tensor = get_random_test_tensor(size=size, is_float=False)
            t_cuda = CUDALongTensor(tensor)
            for dim in range(tensor.dim()):
                reference = tensor.unbind(dim)
                result = torch.unbind(t_cuda, dim)
                result2 = t_cuda.unbind(dim)

                for i in range(len(result)):
                    self.assertTrue(
                        type(result[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self.assertTrue(
                        type(result2[i]) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self._check_int(
                        result[i].cpu(), reference[i], "unbind failed on CUDALongTensor"
                    )
                    self._check_int(
                        result2[i].cpu(),
                        reference[i],
                        "unbind failed on CUDALongTensor",
                    )

    def test_torch_gather(self) -> None:
        """Test torch.gather on CUDALongTensor"""
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for size in sizes:
            for dim in range(len(size)):
                tensor = get_random_test_tensor(size=size, is_float=False)
                index = get_random_test_tensor(size=size, is_float=False)
                index = index.abs().clamp(0, 4)

                t_cuda = CUDALongTensor(tensor)
                idx_cuda = CUDALongTensor(index)

                reference = tensor.gather(dim, index)
                result = t_cuda.gather(dim, idx_cuda)
                result2 = torch.gather(t_cuda, dim, idx_cuda)

                self._check_int(
                    result.cpu(), reference, f"gather failed with size {size}"
                )
                self._check_int(
                    result2.cpu(), reference, f"gather failed with size {size}"
                )

    @unittest.skip("torch.scatter behaves inconsistently on CUDA")
    def test_torch_scatter(self) -> None:
        """Test scatter/scatter_add function of CUDALongTensor

        This test will be skipped for now since torch.scatter provides
        inconsistent result given the same input on CUDA. This is likely
        due to a potential bug on pytorch's implementation of scatter
        """

        funcs = ["scatter", "scatter_add"]
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for func in funcs:
            for size in sizes:
                for dim in range(len(size)):
                    tensor1 = get_random_test_tensor(size=size, is_float=False)
                    tensor2 = get_random_test_tensor(size=size, is_float=False)
                    index = get_random_test_tensor(size=size, is_float=False)
                    index = index.abs().clamp(0, 4)

                    t1_cuda = CUDALongTensor(tensor1)
                    t2_cuda = CUDALongTensor(tensor2)
                    idx_cuda = CUDALongTensor(index)
                    reference = getattr(torch, func)(tensor1, dim, index, tensor2)
                    result = getattr(torch, func)(t1_cuda, dim, idx_cuda, t2_cuda)
                    result2 = getattr(t1_cuda, func)(dim, idx_cuda, t2_cuda)

                    self.assertTrue(
                        type(result) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self.assertTrue(
                        type(result2) == CUDALongTensor,
                        "result should be a CUDALongTensor",
                    )
                    self._check_int(result.cpu(), reference, "{} failed".format(func))
                    self._check_int(result2.cpu(), reference, "{} failed".format(func))

    def test_torch_nonzero(self) -> None:
        """Test torch.nonzero on CUDALongTensor"""
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for size in sizes:
            t1 = get_random_test_tensor(size=size, is_float=False)
            t1_cuda = CUDALongTensor(t1)

            ref = t1.nonzero(as_tuple=False)
            ref_tuple = t1.nonzero(as_tuple=True)

            result = t1_cuda.nonzero(as_tuple=False)
            result_tuple = t1_cuda.nonzero(as_tuple=True)

            self.assertTrue(
                type(result) == CUDALongTensor, "result should be a CUDALongTensor"
            )
            self._check_int(result.cpu(), ref, "nonzero failed")
            for i in range(len(result_tuple)):
                self.assertTrue(
                    type(result_tuple[i]) == CUDALongTensor,
                    "result should be a CUDALongTensor",
                )
                self._check_int(result_tuple[i].cpu(), ref_tuple[i], "nonzero failed")

    @unittest.skip("torch.scatter behaves inconsistently on CUDA")
    def test_scatter(self) -> None:
        """This test will be skipped for now since torch.scatter provides
        inconsistent result given the same input on CUDA. This is likely
        due to a potential bug on pytorch's implementation of scatter
        """
        pass


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestCUDA):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.device = torch.device("cuda")

    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestCUDA):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.device = torch.device("cuda")

    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        crypten.CrypTensor.set_grad_enabled(False)
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target of another test)
if __name__ == "__main__":
    unittest.main()
