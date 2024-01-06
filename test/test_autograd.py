#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import unittest

import crypten
import crypten.gradients as gradients
import torch
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.gradients import AutogradContext, AutogradFunction
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase


class TestAutograd:
    """
    This class tests all autograd-related functionality.
    """

    def setUp(self):
        # we do not want main process (rank -1) initializing the communicator:
        if self.rank >= 0:
            crypten.init()

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # check that sizes match:
        self.assertTrue(tensor.size() == reference.size(), msg)

        # check that values match:
        if is_float_tensor(reference):
            diff = (tensor - reference).abs_()
            norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
            test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
            test_passed = test_passed.gt(0).all().item() == 1
        else:
            test_passed = (tensor == reference).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def test_non_differentiable_marking(self):
        """Tests whether marking of non-differentiability works correctly."""

        # generate random inputs:
        inputs = [get_random_test_tensor(is_float=True) for _ in range(5)]
        inputs = [crypten.cryptensor(input) for input in inputs]
        ctx = AutogradContext()

        # repeat test multiple times:
        for _ in range(10):

            # mark non-differentiable inputs as such:
            differentiable = [random.random() > 0.5 for _ in range(len(inputs))]
            for idx, diff in enumerate(differentiable):
                if not diff:
                    ctx.mark_non_differentiable(inputs[idx])

            # check that inputs were correctly marked:
            for idx, input in enumerate(inputs):
                self.assertEqual(
                    ctx.is_differentiable(input),
                    differentiable[idx],
                    "marking of differentiability failed",
                )
            ctx.reset()

        # test behavior of autograd in CrypTensor:
        input = inputs[0]
        input.requires_grad = True
        reference = [True, True, False]
        for func_name in ["min", "max"]:
            outputs = [None] * 3
            outputs[0] = getattr(input, func_name)()
            outputs[1], outputs[2] = getattr(input, func_name)(0)
            for idx, output in enumerate(outputs):
                self.assertEqual(
                    output.requires_grad,
                    reference[idx],
                    "value of requires_grad is incorrect",
                )

        # behavior of max_pool2d in which indices are returned:
        input = get_random_test_tensor(size=(1, 3, 8, 8), is_float=True)
        input = crypten.cryptensor(input, requires_grad=True)
        reference = [True, True, False]
        outputs = [None] * 3
        outputs[0] = input.max_pool2d(2, return_indices=False)
        outputs[1], outputs[2] = input.max_pool2d(2, return_indices=True)
        for idx, output in enumerate(outputs):
            self.assertEqual(
                output.requires_grad,
                reference[idx],
                "value of requires_grad is incorrect",
            )

    def test_inplace(self):
        """
        Tests that in-place functions cannot be used in autograd but return
        correct results outside of autograd.
        """
        value = 1.5
        reference = get_random_test_tensor(size=(1, 3, 8, 8), is_float=True)
        for requires_grad in [False, True]:
            result = crypten.cryptensor(reference, requires_grad=requires_grad)
            if requires_grad:
                with self.assertRaises(RuntimeError):
                    result.add_(value)
            else:
                result.add_(value)
                self._check(result, reference.add(value), "in-place addition failed")

    def test_autograd_registation(self):
        """Tests registration of new autograd function."""

        # check that get_grad_fn() returns correct functions:
        for func_name, reference_func in gradients.FUNCTION_REGISTRY.items():
            grad_fn = gradients.get_grad_fn(func_name)
            self.assertEqual(grad_fn, reference_func)
            self.assertEqual(grad_fn.name, func_name)

        # check that non-existing functions return None:
        for invalid_func_name in ["bfobofb", "djhfhr"]:
            func = gradients.get_grad_fn(invalid_func_name)
            self.assertIsNone(func)

        # check that registering new classes works:
        for func_name in ["mock_func1", "mock_func2", "mock_func3"]:
            cls = type("%sName" % func_name, (AutogradFunction,), {})
            gradients.register_function(func_name)(cls)
            grad_fn = gradients.get_grad_fn(func_name)
            self.assertEqual(grad_fn, cls)
            self.assertEqual(grad_fn.name, func_name)

        # check that existing functions cannot be overwritten:
        for func_name in ["add", "sub", "view"]:
            cls = type("%sName" % func_name, (AutogradFunction,), {})
            with self.assertRaises(ValueError):
                gradients.register_function(func_name)(cls)

    def test_autograd_func_take(self):
        """Tests the part of autograd take that does not have a torch equivalent"""
        tensor_size = [5, 5, 5, 5]
        index = torch.tensor([[[1, 2], [3, 4]], [[4, 2], [1, 3]]], dtype=torch.long)

        # Test when dimension!=None
        for dimension in range(0, 4):
            tensor = get_random_test_tensor(size=tensor_size, is_float=True)
            ref_forward = torch.from_numpy(tensor.numpy().take(index, dimension))
            encrypted_tensor = crypten.cryptensor(tensor)
            encr_inputs = [encrypted_tensor, index, dimension]

            # test forward
            ctx = AutogradContext()
            grad_fn_take = gradients.get_grad_fn("take")
            encr_output = grad_fn_take.forward(ctx, *encr_inputs)
            self._check(encr_output, ref_forward, "take forward failed: dimension set")

            # test backward:
            # first, recreate take forward function with only torch operations
            tensor2 = get_random_test_tensor(size=tensor_size, is_float=True)
            tensor2.requires_grad = True
            all_indices = [slice(0, x) for x in tensor2.size()]
            all_indices[dimension] = index
            ref_forward_torch = tensor2[all_indices]
            grad_output = torch.ones(ref_forward_torch.size())
            ref_forward_torch.backward(grad_output)

            # next, do backward pass on encrypted tensor
            encr_grad_output = encr_output.new(grad_output)
            encr_grad = grad_fn_take.backward(ctx, encr_grad_output)

            # finally, compare values
            self._check(encr_grad, tensor2.grad, "take backward failed: dimension set")

    def test_detach(self):
        """Tests that detach() works as expected."""

        for func_name in ["detach", "detach_"]:

            # get test case:
            input_size = (12, 5)
            input1 = get_random_test_tensor(size=input_size, is_float=True)
            input2 = get_random_test_tensor(size=input_size, is_float=True)
            input1 = crypten.cryptensor(input1, requires_grad=True)
            input2 = crypten.cryptensor(input2, requires_grad=True)

            # perform forward computation with detach in the middle:
            intermediate = input1.add(1.0)
            intermediate = getattr(intermediate, func_name)()
            output = intermediate.add(input2).sum()

            # perform backward:
            output.backward()
            msg = "detach() function does not behave as expected"
            self.assertIsNone(output.grad, msg)
            self.assertIsNone(intermediate.grad, msg)
            self.assertIsNone(input1.grad, msg)
            self.assertIsNotNone(input2.grad, msg)

    def test_forward_tracking(self):
        """Tests that requires_grad influences tracking of forward computations."""

        for requires_grad in [True, False]:

            # get test case:
            input = get_random_test_tensor(size=(12, 5), is_float=True)
            input = crypten.cryptensor(input, requires_grad=requires_grad)

            # perform forward computation:
            output = input.exp().sum()

            # setting requires_grad post-hoc should not affect backward behavior:
            input.requires_grad = True
            output.requires_grad = True
            output.backward()

            # check results:
            msg = "tracking of forward computations does not work as expected"
            if requires_grad:
                self.assertIsNotNone(input.grad, msg)
                self.assertIsNone(output.grad, msg)
            else:
                self.assertIsNone(input.grad, msg)
                self.assertIsNotNone(output.grad, msg)

    def test_autograd_accumulation(self):
        """Tests accumulation in autograd."""

        # graphs that have nodes with multiple parents, dead leafs, etc.:
        def test_case1(input, encr_input):
            output = input.add(1.0).add(input.exp()).sum()
            encr_output = encr_input.add(1.0).add(encr_input.exp()).sum()
            return output, encr_output

        def test_case2(input, encr_input):
            intermediate = input.pow(2.0)  # PyTorch
            output = intermediate.add(1.0).add(intermediate.mul(2.0)).sum()
            encr_intermediate = encr_input.square()  # CrypTen
            encr_output = (
                encr_intermediate.add(1.0).add(encr_intermediate.mul(2.0)).sum()
            )
            return output, encr_output

        def test_case3(input, encr_input):
            intermediate1 = input.pow(2.0)  # PyTorch
            intermediate2 = intermediate1.add(1.0).add(intermediate1.mul(2.0))
            output = intermediate2.pow(2.0).sum()
            encr_intermediate1 = encr_input.square()  # CrypTen
            encr_intermediate2 = encr_intermediate1.add(1.0).add(
                encr_intermediate1.mul(2.0)
            )
            encr_output = encr_intermediate2.square().sum()
            return output, encr_output

        def test_case4(input, encr_input):
            intermediate1 = input.mul(3.0).add(2.0).pow(2.0)  # PyTorch
            intermediate2 = intermediate1.add(1.0).add(intermediate1.mul(2.0))
            output = intermediate2.pow(2.0).sum()
            encr_intermediate1 = encr_input.mul(3.0).add(2.0).square()  # CrypTen
            encr_intermediate2 = encr_intermediate1.add(1.0).add(
                encr_intermediate1.mul(2.0)
            )
            encr_output = encr_intermediate2.square().sum()
            return output, encr_output

        def test_case5(input, encr_input):
            intermediate1 = input.mul(3.0)  # PyTorch
            intermediate2 = input.add(2.0).pow(2.0)
            intermediate3 = input.pow(2.0)
            output = (
                torch.cat([intermediate1, intermediate2, intermediate3]).mul(0.5).sum()
            )
            encr_intermediate1 = encr_input.mul(3.0)  # CrypTen
            encr_intermediate2 = encr_input.add(2.0).square()
            encr_intermediate3 = encr_input.pow(2.0)
            encr_output = (
                crypten.cat(
                    [encr_intermediate1, encr_intermediate2, encr_intermediate3]
                )
                .mul(0.5)
                .sum()
            )
            return output, encr_output

        def test_case6(input, encr_input):
            idx1 = torch.tensor([[0, 2, 4, 3, 8]], dtype=torch.long)
            idx2 = torch.tensor([[5, 1, 3, 5, 2]], dtype=torch.long)
            idx3 = torch.tensor([[2, 3, 1]], dtype=torch.long)
            intermediate1 = input.gather(0, idx1).gather(1, idx3).pow(2.0)  # PyTorch
            intermediate2 = input.gather(0, idx2).gather(1, idx3).add(-2.0)
            output = torch.cat([intermediate1, intermediate2]).mul(0.5).sum()
            encr_intermediate1 = (
                encr_input.gather(0, idx1).gather(1, idx3).square()
            )  # CrypTen
            encr_intermediate2 = encr_input.gather(0, idx2).gather(1, idx3).add(-2.0)
            encr_output = (
                crypten.cat([encr_intermediate1, encr_intermediate2], dim=0)
                .mul(0.5)
                .sum()
            )
            return output, encr_output

        def test_case7(input, encr_input):
            intermediate1 = input.add(3.0)  # PyTorch
            intermediate2 = input.add(2.0).pow(2.0)
            intermediate3 = intermediate1.add(intermediate2)
            intermediate4 = intermediate1.add(intermediate2)
            output = intermediate3.add(intermediate4).sum()
            encr_intermediate1 = encr_input.add(3.0)  # CrypTen
            encr_intermediate2 = encr_input.add(2.0).pow(2.0)
            encr_intermediate3 = encr_intermediate1.add(encr_intermediate2)
            encr_intermediate4 = encr_intermediate1.add(encr_intermediate2)
            encr_output = encr_intermediate3.add(encr_intermediate4).sum()
            return output, encr_output

        def test_case8(input, encr_input):
            intermediate1 = input.add(3.0)
            intermediate2 = torch.cat([input, intermediate1])
            intermediate3 = intermediate2.pow(2.0)
            output = torch.cat([input, intermediate2, intermediate3]).add(-1).sum()

            encr_intermediate1 = encr_input.add(3.0)
            encr_intermediate2 = crypten.cat([encr_input, encr_intermediate1])
            encr_intermediate3 = encr_intermediate2.pow(2.0)
            encr_output = (
                crypten.cat([encr_input, encr_intermediate2, encr_intermediate3])
                .add(-1)
                .sum()
            )

            return output, encr_output

        def test_case9(input, encr_input):
            intermediate1 = torch.cat([input, input])
            intermediate2 = intermediate1.mean(0, keepdim=True)
            output = torch.cat([intermediate2, intermediate1], dim=0).sum()

            encr_intermediate1 = crypten.cat([encr_input, encr_input])
            encr_intermediate2 = encr_intermediate1.mean(0, keepdim=True)
            encr_output = crypten.cat([encr_intermediate2, encr_intermediate1]).sum()

            return output, encr_output

        # loop over test cases:
        test_cases = [
            value
            for key, value in locals().items()
            if callable(value) and key.startswith("test_case")
        ]
        for idx, test_case in enumerate(test_cases):

            # get input tensors:
            input = get_random_test_tensor(size=(12, 5), is_float=True)
            input.requires_grad = True
            encr_input = crypten.cryptensor(input, requires_grad=True)

            # perform multiple forward computations on input that get combined:
            output, encr_output = test_case(input, encr_input)
            self._check(
                encr_output._tensor, output, "forward for test case %d failed" % idx
            )
            self.assertTrue(
                encr_output.requires_grad,
                "requires_grad incorrect for test case %d" % idx,
            )

            # perform backward computation:
            output.backward()
            encr_output.backward()
            self._check(
                encr_input.grad, input.grad, "backward for test case %d failed" % idx
            )

        # test cases in which tensor gets combined with itself:
        for func_name in ["sub", "add", "mul"]:

            # get input tensors:
            input = get_random_test_tensor(size=(12, 5), is_float=True)
            input.requires_grad = True
            encr_input = crypten.cryptensor(input, requires_grad=True)

            # perform forward-backward pass:
            output = getattr(input, func_name)(input).sum()
            encr_output = getattr(encr_input, func_name)(encr_input).sum()
            self._check(encr_output._tensor, output, "forward failed")
            self.assertTrue(encr_output.requires_grad, "requires_grad incorrect")
            output.backward()
            encr_output.backward()
            self._check(encr_input.grad, input.grad, "%s backward failed" % func_name)

    def test_autograd(self) -> None:
        """Tests autograd graph construction and backprop."""

        # define test cases:
        tests = [
            (1, ["relu", "neg", "relu", "sum"]),
            (2, ["t", "neg", "add", "sum"]),
            (2, ["relu", "mul", "t", "sum"]),
        ]
        binary_functions = ["add", "sub", "mul", "dot", "matmul"]

        # PyTorch test case:
        for test in tests:

            # get test case:
            number_of_inputs, ops = test
            inputs = [
                get_random_test_tensor(size=(12, 5), is_float=True)
                for _ in range(number_of_inputs)
            ]
            encr_inputs = [crypten.cryptensor(input) for input in inputs]

            # get autograd variables:
            for input in inputs:
                input.requires_grad = True
            for encr_input in encr_inputs:
                encr_input.requires_grad = True

            # perform forward pass, logging all intermediate outputs:
            outputs, encr_outputs = [inputs], [encr_inputs]
            for op in ops:

                # get inputs for current operation:
                input, output = outputs[-1], []
                encr_input, encr_output = encr_outputs[-1], []

                # apply current operation:
                if op in binary_functions:  # combine outputs via operation
                    output.append(getattr(input[0], op)(input[1]))
                    encr_output.append(getattr(encr_input[0], op)(encr_input[1]))
                else:
                    for idx in range(len(input)):
                        output.append(getattr(input[idx], op)())
                        encr_output.append(getattr(encr_input[idx], op)())

                # keep references to outputs of operation:
                outputs.append(output)
                encr_outputs.append(encr_output)

            # check output of forward pass:
            output, encr_output = outputs[-1][0], encr_outputs[-1][0]
            self._check(encr_output._tensor, output, "forward failed")
            self.assertTrue(encr_output.requires_grad, "requires_grad incorrect")

            # perform backward pass:
            output.backward()
            encr_output.backward()

            # test result of running backward function:
            for idx in range(number_of_inputs):
                self._check(encr_inputs[idx].grad, inputs[idx].grad, "backward failed")

    def test_autograd_repetition(self) -> None:
        """Tests running autograd on the same input repeatedly."""

        # create test case:
        input = get_random_test_tensor(size=(12, 5), is_float=True)
        input.requires_grad = True
        encr_input = crypten.cryptensor(input, requires_grad=True)

        # re-use the same input multiple times:
        for _ in range(7):

            # perform forward pass:
            output = input.exp().sum()
            encr_output = encr_input.exp().sum()
            self._check(encr_output._tensor, output, "forward failed")
            self.assertTrue(encr_output.requires_grad, "requires_grad incorrect")

            # perform backward computation:
            output.backward()
            encr_output.backward()
            self._check(encr_input.grad, input.grad, "backward failed")


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestAutograd):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestAutograd):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TTP"
        super(TestTTP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
