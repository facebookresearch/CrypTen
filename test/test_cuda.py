#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import unittest
from test.multiprocess_test_case import get_random_test_tensor

import crypten
import torch
import torch.nn.functional as F
from crypten.common.tensor_types import is_float_tensor
from crypten.cuda import CUDALongTensor
from crypten.mpc import MPCTensor


@unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
class TestCUDA(unittest.TestCase):
    """
        This class tests all functions of CUDALongTensor as well as its integration with MPCTensor.
    """

    def __init__(self, methodName):
        super().__init__(methodName)
        crypten.init()

    def _check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text(dst=dst)
        if dst is not None and dst != self.rank:
            self.assertIsNone(tensor)
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")

        if tensor.device != reference.device:
            tensor = tensor.cpu()
            reference = reference.cpu()

        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Reference %s" % reference)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def _check_tuple(self, encrypted_tuple, reference, msg, tolerance=None):
        self.assertTrue(isinstance(encrypted_tuple, tuple))
        self.assertEqual(len(encrypted_tuple), len(reference))
        for i in range(len(reference)):
            self._check(encrypted_tuple[i], reference[i], msg, tolerance=tolerance)

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

    def test_patched_matmul(self):
        x = get_random_test_tensor(max_value=2 ** 62, is_float=False)
        x_cuda = CUDALongTensor(x.cuda())
        for width in range(2, x.nelement()):
            matrix_size = (x.nelement(), width)
            y = get_random_test_tensor(
                size=matrix_size, max_value=2 ** 62, is_float=False
            )

            y_cuda = CUDALongTensor(y.cuda())
            z = torch.matmul(x_cuda, y_cuda).data

            reference = torch.matmul(x, y)
            self._check_int(z.cpu(), reference, "matmul failed for cuda_patches")

    def test_conv1d_smaller_signal_one_channel(self):
        self._patched_conv1d(5, 1)

    def test_conv1d_smaller_signal_many_channels(self):
        self._patched_conv1d(5, 5)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_larger_signal_one_channel(self):
        self._patched_conv1d(16, 1)

    @unittest.skipIf(torch.cuda.is_available() is False, "requires CUDA")
    def test_conv1d_larger_signal_many_channels(self):
        self._patched_conv1d(16, 5)

    def _conv1d(self, signal_size, in_channels):
        """Test convolution of encrypted tensor with public/private tensors."""
        nbatches = [1, 3]
        kernel_sizes = [1, 2, 3]
        ochannels = [1, 3, 6]
        paddings = [0, 1]
        strides = [1, 2]

        for func_name in ["conv1d", "conv_transpose1d"]:
            for kernel_type in [lambda x: x, MPCTensor]:
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
                    signal = get_random_test_tensor(
                        size=input_size, is_float=True
                    ).cuda()

                    if func_name == "conv1d":
                        k_size = (out_channels, in_channels, kernel_size)
                    else:
                        k_size = (in_channels, out_channels, kernel_size)
                    kernel = get_random_test_tensor(size=k_size, is_float=True).cuda()

                    reference = getattr(F, func_name)(
                        signal, kernel, padding=padding, stride=stride
                    )
                    encrypted_signal = MPCTensor(signal)
                    encrypted_kernel = kernel_type(kernel)
                    encrypted_conv = getattr(encrypted_signal, func_name)(
                        encrypted_kernel, padding=padding, stride=stride
                    )

                    self._check(encrypted_conv, reference, f"{func_name} failed")

    def _patched_conv1d(self, signal_size, in_channels):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
        nbatches = [1, 3]
        kernel_sizes = [1, 2, 3]
        ochannels = [1, 3, 6]
        paddings = [0, 1]
        strides = [1, 2]

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

                signal_cuda = CUDALongTensor(signal.cuda())
                kernel_cuda = CUDALongTensor(kernel.cuda())

                reference = getattr(F, func_name)(
                    signal, kernel, padding=padding, stride=stride
                )
                result = getattr(F, func_name)(
                    signal_cuda, kernel_cuda, padding=padding, stride=stride
                )
                result = result.data.cpu()

                self._check_int(result, reference, f"{func_name} failed")

    def test_conv2d_square_image_one_channel(self):
        self._patched_conv2d((5, 5), 1)

    def test_conv2d_square_image_many_channels(self):
        self._patched_conv2d((5, 5), 5)

    def test_conv2d_rectangular_image_one_channel(self):
        self._patched_conv2d((16, 7), 1)

    def test_conv2d_rectangular_image_many_channels(self):
        self._patched_conv2d((16, 7), 5)

    def _patched_conv2d(self, image_size, in_channels):
        """Test convolution of torch.cuda.LongTensor with cuda_patches technique."""
        nbatches = [1, 3]
        kernel_sizes = [(1, 1), (2, 2), (2, 3)]
        ochannels = [1, 3, 6]
        paddings = [0, 1, (0, 1)]
        strides = [1, 2, (1, 2)]

        for func_name in ["conv2d"]:
            for (
                batches,
                kernel_size,
                out_channels,
                padding,
                stride,
            ) in itertools.product(
                nbatches, kernel_sizes, ochannels, paddings, strides
            ):

                # sample input:
                input_size = (batches, in_channels, *image_size)
                input = get_random_test_tensor(size=input_size, is_float=False)

                # sample filtering kernel:
                if func_name == "conv2d":
                    k_size = (out_channels, in_channels, *kernel_size)
                else:
                    k_size = (in_channels, out_channels, *kernel_size)
                kernel = get_random_test_tensor(size=k_size, is_float=False)

                input_cuda = CUDALongTensor(input.cuda())
                kernel_cuda = CUDALongTensor(kernel.cuda())

                result = getattr(F, func_name)(
                    input_cuda, kernel_cuda, padding=padding, stride=stride
                )
                result = result.data.cpu()

                # check that result is correct:
                reference = getattr(F, func_name)(
                    input, kernel, padding=padding, stride=stride
                )
                self._check_int(result, reference, "%s failed" % func_name)

    def test_torch_arithmetic(self):
        """Test torch arithmetic on CUDALongTensor"""
        funcs = ["add", "sub", "mul", "div"]
        a = get_random_test_tensor(is_float=False)
        b = get_random_test_tensor(is_float=False)

        a_cuda = CUDALongTensor(a)
        b_cuda = CUDALongTensor(b)

        for op in funcs:
            reference = getattr(torch, op)(a, b)
            result = getattr(torch, op)(a_cuda, b_cuda)
            result2 = getattr(a_cuda, op)(b_cuda)

            self.assertTrue(type(result), CUDALongTensor)
            self._check_int(
                reference, result.cpu(), "torch.{} failed for CUDALongTensor".format(op)
            )
            self._check_int(
                reference,
                result2.cpu(),
                "torch.{} failed for CUDALongTensor".format(op),
            )

    def test_torch_comparators(self):
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

    def test_torch_stack_cat(self):
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

    def test_torch_broadcast_tensor(self):
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

    def test_torch_split(self):
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

    def test_torch_unbind(self):
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

    def test_torch_gather(self):
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
    def test_torch_scatter(self):
        """ Test scatter/scatter_add function of CUDALongTensor

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

    def test_torch_nonzero(self):
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
