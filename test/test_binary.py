#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import unittest

import crypten
import torch
from crypten.common.tensor_types import is_int_tensor
from crypten.mpc.primitives import BinarySharedTensor
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestBinary(MultiProcessTestCase):
    """
    This class tests all functions of BinarySharedTensor.
    """

    def setUp(self) -> None:
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
        if self.rank >= 0:
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

        self.assertTrue(is_int_tensor(reference), "reference must be a long")
        test_passed = (tensor == reference).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def test_encrypt_decrypt(self) -> None:
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

            # encryption and decryption without source:
            reference = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor = BinarySharedTensor(reference)
            self._check(encrypted_tensor, reference, "en/decryption failed")
            for dst in range(self.world_size):
                self._check(
                    encrypted_tensor, reference, "en/decryption failed", dst=dst
                )

            # encryption and decryption with source:
            for src in range(self.world_size):
                input_tensor = reference if src == self.rank else []
                encrypted_tensor = BinarySharedTensor(
                    input_tensor, src=src, broadcast_size=True
                )
                for dst in range(self.world_size):
                    self._check(
                        encrypted_tensor,
                        reference,
                        "en/decryption with broadcast_size failed",
                        dst=dst,
                    )

    def test_transpose(self) -> None:
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
            tensor = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor = BinarySharedTensor(tensor)

            if len(size) == 2:  # t() asserts dim == 2
                reference = tensor.t()
                encrypted_out = encrypted_tensor.t()
                self._check(encrypted_out, reference, "t() failed")

            for dim0 in range(len(size)):
                for dim1 in range(len(size)):
                    reference = tensor.transpose(dim0, dim1)
                    encrypted_out = encrypted_tensor.transpose(dim0, dim1)
                    self._check(encrypted_out, reference, "transpose failed")

    def test_permute(self) -> None:
        """Test the permute operations"""
        sizes = [
            (1,),
            (5,),
            (1, 5),
            (1, 5, 7),
            (7, 1, 5),
            (5, 7, 1),
            (1, 3, 5, 7),
            (5, 3, 32, 32),
        ]
        for size in sizes:
            tensor = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor = BinarySharedTensor(tensor)

            # test reversing the dimensions
            dim_arr = [x - 1 for x in range(tensor.dim(), 0, -1)]
            reference = tensor.permute(dim_arr)
            encrypted_out = encrypted_tensor.permute(dim_arr)
            self._check(encrypted_out, reference, "permute failed")

            # test one particular non-reversed permutation
            if tensor.dim() == 4:
                dim_arr = [1, 3, 0, 2]
                reference = tensor.permute(dim_arr)
                encrypted_out = encrypted_tensor.permute(dim_arr)
                self._check(encrypted_out, reference, "permute failed")

    def test_XOR(self) -> None:
        """Test bitwise-XOR function on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor ^ tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            encrypted_out = encrypted_tensor ^ encrypted_tensor2
            self._check(encrypted_out, reference, "%s XOR failed" % tensor_type)

    def test_AND(self) -> None:
        """Test bitwise-AND function on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor & tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            encrypted_out = encrypted_tensor & encrypted_tensor2
            self._check(encrypted_out, reference, "%s AND failed" % tensor_type)

    def test_OR(self) -> None:
        """Test bitwise-OR function on BinarySharedTensor"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor | tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            encrypted_out = encrypted_tensor | encrypted_tensor2
            self._check(encrypted_out, reference, "%s OR failed" % tensor_type)

    def test_bitwise_broadcasting(self) -> None:
        """Tests bitwise function broadcasting"""
        bitwise_ops = ["__and__", "__or__", "__xor__"]
        sizes = [
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

        for tensor_type in [lambda x: x, BinarySharedTensor]:
            for op in bitwise_ops:
                for size1, size2 in itertools.combinations(sizes, 2):
                    tensor1 = get_random_test_tensor(size=size1, is_float=False)
                    tensor2 = get_random_test_tensor(size=size2, is_float=False)
                    encrypted_tensor1 = BinarySharedTensor(tensor1)
                    tensor2_transformed = tensor_type(tensor2)

                    if isinstance(tensor2_transformed, BinarySharedTensor):
                        tensor2_transformed_type = "private"
                    else:
                        tensor2_transformed_type = "public"

                    self._check(
                        getattr(encrypted_tensor1, op)(tensor2_transformed),
                        getattr(tensor1, op)(tensor2),
                        f"{tensor2_transformed_type} {op} broadcasting "
                        f"failed with sizes {size1}, {size2}",
                    )

    def test_invert(self) -> None:
        """Test bitwise-invert function on BinarySharedTensor"""
        tensor = get_random_test_tensor(is_float=False)
        encrypted_tensor = BinarySharedTensor(tensor)
        reference = ~tensor
        encrypted_out = ~encrypted_tensor
        self._check(encrypted_out, reference, "invert failed")

    def test_add(self) -> None:
        """Tests add using binary shares"""
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            tensor = get_random_test_tensor(is_float=False)
            tensor2 = get_random_test_tensor(is_float=False)
            reference = tensor + tensor2
            encrypted_tensor = BinarySharedTensor(tensor)
            encrypted_tensor2 = tensor_type(tensor2)
            encrypted_out = encrypted_tensor + encrypted_tensor2
            self._check(encrypted_out, reference, "%s add failed" % tensor_type)

    def test_comparators(self) -> None:
        """Test comparators (>, >=, <, <=, ==, !=)"""
        for _scale in [False, True]:
            for comp in ["gt", "ge", "lt", "le", "eq", "ne"]:
                for tensor_type in [lambda x: x, BinarySharedTensor]:
                    tensor = get_random_test_tensor(is_float=False)
                    tensor2 = get_random_test_tensor(is_float=False)

                    encrypted_tensor = BinarySharedTensor(tensor)
                    encrypted_tensor2 = tensor_type(tensor2)

                    reference = getattr(tensor, comp)(tensor2).long()
                    encrypted_out = getattr(encrypted_tensor, comp)(encrypted_tensor2)

                    self._check(encrypted_out, reference, "%s comparator failed" % comp)

    def test_sum(self) -> None:
        """Tests sum using binary shares"""
        tensor = get_random_test_tensor(size=(5, 5, 5), is_float=False)
        encrypted = BinarySharedTensor(tensor)
        self._check(encrypted.sum(), tensor.sum(), "sum failed")

        for dim in [0, 1, 2]:
            reference = tensor.sum(dim)
            encrypted_out = encrypted.sum(dim)
            self._check(encrypted_out, reference, "sum failed")

    def test_get_set(self) -> None:
        for tensor_type in [lambda x: x, BinarySharedTensor]:
            for size in range(1, 5):
                # Test __getitem__
                tensor = get_random_test_tensor(size=(size, size), is_float=False)
                reference = tensor[:, 0]

                encrypted_tensor = BinarySharedTensor(tensor)
                encrypted_out = encrypted_tensor[:, 0]
                self._check(encrypted_out, reference, "getitem failed")

                reference = tensor[0, :]
                encrypted_out = encrypted_tensor[0, :]
                self._check(encrypted_out, reference, "getitem failed")

                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,), is_float=False)
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = BinarySharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = BinarySharedTensor(tensor)
                encrypted2 = tensor_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference, "%s setitem failed" % type(encrypted2)
                )

    def test_share_attr(self) -> None:
        """Tests share attribute getter and setter"""
        for is_float in (True, False):
            reference = get_random_test_tensor(is_float=is_float)
            encrypted_tensor = BinarySharedTensor(reference)
            self.assertTrue(
                torch.equal(encrypted_tensor.share, encrypted_tensor.share),
                "share getter failed",
            )

            new_share = get_random_test_tensor(is_float=False)
            encrypted_tensor.share = new_share
            self.assertTrue(
                torch.equal(encrypted_tensor.share, new_share), "share setter failed"
            )

    def test_inplace(self) -> None:
        """Test inplace vs. out-of-place functions"""
        for op in ["__xor__", "__and__", "__or__"]:
            for tensor_type in [lambda x: x, BinarySharedTensor]:
                tensor1 = get_random_test_tensor(is_float=False)
                tensor2 = get_random_test_tensor(is_float=False)

                reference = getattr(tensor1, op)(tensor2)

                encrypted1 = BinarySharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                input_plain_id = id(encrypted1.share)
                input_encrypted_id = id(encrypted1)

                # Test that out-of-place functions do not modify the input
                private = isinstance(encrypted2, BinarySharedTensor)
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
                inplace_op = op[:2] + "i" + op[2:]
                encrypted_out = getattr(encrypted1, inplace_op)(encrypted2)
                self._check(
                    encrypted1,
                    reference,
                    "%s in-place %s does not modify input"
                    % ("private" if private else "public", inplace_op),
                )
                self._check(
                    encrypted_out,
                    reference,
                    "%s in-place %s produces incorrect output"
                    % ("private" if private else "public", inplace_op),
                )
                self.assertTrue(id(encrypted_out.share) == input_plain_id)
                self.assertTrue(id(encrypted_out) == input_encrypted_id)

    def test_control_flow_failure(self) -> None:
        """Tests that control flow fails as expected"""
        tensor = get_random_test_tensor(is_float=False)
        encrypted_tensor = BinarySharedTensor(tensor)
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

    def test_src_failure(self) -> None:
        """Tests that out-of-bounds src fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        for src in [None, "abc", -2, self.world_size]:
            with self.assertRaises(AssertionError):
                BinarySharedTensor(tensor, src=src)

    def test_src_match_input_data(self) -> None:
        """Tests incorrect src in BinarySharedTensor fails as expected"""
        tensor = get_random_test_tensor(is_float=True)
        tensor.src = 0
        for testing_src in [None, "abc", -2, self.world_size]:
            with self.assertRaises(AssertionError):
                BinarySharedTensor(tensor, src=testing_src)

    def test_where(self) -> None:
        """Tests where() conditional element selection"""
        sizes = [(10,), (5, 10), (1, 5, 10)]
        y_types = [lambda x: x, BinarySharedTensor]

        for size, y_type in itertools.product(sizes, y_types):
            tensor1 = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor1 = BinarySharedTensor(tensor1)
            tensor2 = get_random_test_tensor(size=size, is_float=False)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                get_random_test_tensor(max_value=1, size=[1], is_float=False) + 1
            )
            condition_encrypted = BinarySharedTensor(condition_tensor)
            condition_bool = condition_tensor.bool()

            reference_out = tensor1 * condition_tensor + tensor2 * (
                1 - condition_tensor
            )

            encrypted_out = encrypted_tensor1.where(condition_bool, encrypted_tensor2)

            y_is_private = y_type == BinarySharedTensor
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

    def test_gather(self) -> None:
        """Test gather function of encrypted tensor"""
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for size in sizes:
            for dim in range(len(size)):
                tensor = get_random_test_tensor(size=size, is_float=False)
                index = get_random_test_tensor(size=size, is_float=False)
                index = index.abs().clamp(0, 4)
                encrypted = BinarySharedTensor(tensor)
                reference = tensor.gather(dim, index)
                encrypted_out = encrypted.gather(dim, index)
                self._check(encrypted_out, reference, f"gather failed with size {size}")

    def test_scatter(self) -> None:
        """Test scatter function of encrypted tensor"""
        funcs = ["scatter", "scatter_"]
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]
        for func in funcs:
            for size in sizes:
                for tensor_type in [lambda x: x, BinarySharedTensor]:
                    for dim in range(len(size)):
                        tensor1 = get_random_test_tensor(size=size, is_float=False)
                        tensor2 = get_random_test_tensor(size=size, is_float=False)
                        index = get_random_test_tensor(size=size, is_float=False)
                        index = index.abs().clamp(0, 4)
                        encrypted = BinarySharedTensor(tensor1)
                        encrypted2 = tensor_type(tensor2)
                        reference = getattr(tensor1, func)(dim, index, tensor2)
                        encrypted_out = getattr(encrypted, func)(dim, index, encrypted2)
                        private = tensor_type == BinarySharedTensor
                        self._check(
                            encrypted_out,
                            reference,
                            "%s %s failed" % ("private" if private else "public", func),
                        )
                        if func.endswith("_"):
                            # Check in-place scatter modified input
                            self._check(
                                encrypted,
                                reference,
                                "%s %s failed to modify input"
                                % ("private" if private else "public", func),
                            )
                        else:
                            # Check original is not modified
                            self._check(
                                encrypted,
                                tensor1,
                                "%s %s unintendedly modified input"
                                % ("private" if private else "public", func),
                            )

    def test_split(self) -> None:
        """Test gather function of encrypted tensor"""
        sizes = [(5, 5), (5, 5, 5), (5, 5, 5, 5)]

        for size in sizes:
            for dim in range(len(size)):
                tensor = get_random_test_tensor(size=size, is_float=False)
                encrypted = BinarySharedTensor(tensor)

                for idx in range(6):
                    split = (idx, 5 - idx)
                    reference0, reference1 = tensor.split(split, dim=dim)
                    encrypted_out0, encrypted_out1 = encrypted.split(split, dim=dim)

                    self._check(
                        encrypted_out0, reference0, f"split failed with input {split}"
                    )
                    self._check(
                        encrypted_out1, reference1, f"split failed with input {split}"
                    )

                split = (5,)
                (reference,) = tensor.split(split, dim=dim)
                (encrypted_out,) = encrypted.split(split, dim=dim)
                self._check(
                    encrypted_out, reference, f"split failed with input {split}"
                )

                with self.assertRaises(RuntimeError):
                    encrypted_out.split((5, 1))


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
