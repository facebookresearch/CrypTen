#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.gradients as gradients
import logging
import torch
import torch.nn.functional as F
import unittest

from crypten.autograd_cryptensor import AutogradContext, AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor
from crypten.gradients import AutogradFunction
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor


class TestAutograd(MultiProcessTestCase):
    """
    This class tests all autograd-related functionality.
    """

    benchmarks_enabled = False

    def setUp(self):
        super().setUp()

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
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

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

    def test_autograd_functions(self):
        """Tests individual autograd functions without testing autograd."""

        # input sizes for tests of autograd functions:
        input_size = {
            "t": (2, 4),
            "transpose": (4, 8, 3),
            "view": (8, 6),
            "reshape": (8, 6),
            "squeeze": (12, 1, 6),
            "unsqueeze": (7, 3),
            "neg": (8, 4),
            "relu": (3, 7),
            "tanh": (4, 3),
            "add": (10, 7),
            "sub": (9, 2),
            "mul": (3, 5),
            "matmul": (7, 7),
            "div": (5, 4),
            "pow": (4, 3),
            "square": (8, 5),
            "sqrt": (5, 6),
            "exp": (5, 2),
            "log": (3, 7),
            "dot": (8,),
            "ger": (12,),
            "sin": (5, 4),
            "cos": (9, 3),
            "abs": (8, 5),
            "norm": (3, 2),  # NOTE: Flaky because sqrt only works for values up to 200.
            "sum": (4, 3),
            "mean": (2, 9),
            "var": (3, 4),
            "max": (6, 7),
            "min": (4, 5),
            "sigmoid": (4, 7),
            "softmax": (10, 5),
            "pad": (6, 3),
            "avg_pool2d": (1, 3, 21, 21),
            "max_pool2d": (1, 3, 21, 21),
            "conv2d": (1, 4, 21, 21),
        }
        additional_args = {
            "transpose": [2, 0],
            "view": [(4, 12)],
            "reshape": [(4, 12)],
            "squeeze": [1],
            "unsqueeze": [1],
            "div": [4.0],
            "pow": [2.0],
            "softmax": [1],
            "pad": [(1, 2, 3, 4)],
            "avg_pool2d": [5],
            "max_pool2d": [3],
            "conv2d": [get_random_test_tensor(size=(2, 4, 3, 3), is_float=True)],
        }
        binary_functions = ["add", "sub", "mul", "dot", "ger", "matmul"]
        positive_only = ["pow", "sqrt", "log"]

        # loop over all autograd functions:
        for func_name in input_size.keys():

            # generate inputs:
            inputs = [
                get_random_test_tensor(size=input_size[func_name], is_float=True)
                for _ in range(2 if func_name in binary_functions else 1)
            ]
            if func_name in positive_only:  # some functions do not take negative values
                inputs = [input.abs().add_(0.001) for input in inputs]
            for input in inputs:
                input.requires_grad = True
            encr_inputs = [crypten.cryptensor(input) for input in inputs]
            number_of_inputs = len(inputs)

            # add additional arguments, encrypting only tensors (if found):
            if func_name in additional_args:
                inputs += additional_args[func_name]
                encr_inputs += additional_args[func_name]
                encr_inputs = [crypten.cryptensor(t) if torch.is_tensor(t) else t
                               for t in encr_inputs]

            # AutogradFunction.forward() does not accept unpacked inputs:
            if len(encr_inputs) == 1:
                encr_inputs = encr_inputs[0]

            # test forward function:
            if hasattr(inputs[0], func_name):  # torch.function()
                reference = getattr(inputs[0], func_name)(*inputs[1:])
            elif hasattr(F, func_name):        # torch.nn.functional.function()
                reference = getattr(F, func_name)(*inputs)
            elif func_name == "square":
                reference = inputs[0].pow(2.0)
            else:
                raise ValueError("unknown PyTorch function: %s" % func_name)
            ctx = AutogradContext()
            grad_fn = gradients.get_grad_fn(func_name)
            encr_output = grad_fn.forward(ctx, encr_inputs)
            self._check(encr_output, reference, "%s forward failed" % func_name)

            # run backward functions:
            grad_output = torch.ones(reference.size())
            encr_grad_output = encr_output.new(grad_output)
            reference.backward(grad_output)
            encr_grad = grad_fn.backward(ctx, encr_grad_output)

            # test result of running backward function:
            if not isinstance(encr_grad, (list, tuple)):
                encr_grad = (encr_grad,)
            for idx in range(number_of_inputs):
                self._check(encr_grad[idx], inputs[idx].grad,
                            "%s backward failed" % func_name)

    def test_detach(self):
        """Tests that detach() works as expected."""

        for func_name in ["detach", "detach_"]:

            # get test case:
            input_size = (12, 5)
            input1 = get_random_test_tensor(size=input_size, is_float=True)
            input2 = get_random_test_tensor(size=input_size, is_float=True)
            input1 = AutogradCrypTensor(crypten.cryptensor(input1))
            input2 = AutogradCrypTensor(crypten.cryptensor(input2))

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

    def test_autograd_accumulation(self):
        """Tests accumulation in autograd."""

        # define test cases that have nodes with multiple parents:
        def test_case1(input, encr_input):
            output = input.add(1.0).add(input.exp()).sum()
            encr_output = encr_input.add(1.0).add(encr_input.exp()).sum()
            return output, encr_output

        def test_case2(input, encr_input):
            intermediate = input.pow(2.0)            # PyTorch
            output = intermediate.add(1.0).add(intermediate.mul(2.0)).sum()
            encr_intermediate = encr_input.square()  # CrypTen
            encr_output = encr_intermediate.add(1.0).add(
                encr_intermediate.mul(2.0)
            ).sum()
            return output, encr_output

        def test_case3(input, encr_input):
            intermediate1 = input.pow(2.0)            # PyTorch
            intermediate2 = intermediate1.add(1.0).add(intermediate1.mul(2.0))
            output = intermediate2.pow(2.0).sum()
            encr_intermediate1 = encr_input.square()  # CrypTen
            encr_intermediate2 = encr_intermediate1.add(1.0).add(
                encr_intermediate1.mul(2.0)
            )
            encr_output = encr_intermediate2.square().sum()
            return output, encr_output

        # loop over test cases:
        for idx, test_case in enumerate([test_case1, test_case2, test_case2]):

            # get input tensors:
            input = get_random_test_tensor(size=(12, 5), is_float=True)
            input.requires_grad = True
            encr_input = AutogradCrypTensor(crypten.cryptensor(input))

            # perform multiple forward computations on input that get combined:
            output, encr_output = test_case(input, encr_input)
            self._check(encr_output._tensor, output,
                        "forward for test case %d failed" % idx)
            self.assertTrue(encr_output.requires_grad,
                            "requires_grad incorrect for test case %d" % idx)

            # perform backward computation:
            output.backward()
            encr_output.backward()
            self._check(encr_input.grad, input.grad,
                        "backward for test case %d failed" % idx)

        # test cases in which tensor gets combined with itself:
        for func_name in ["sub", "add", "mul"]:

            # get input tensors:
            input = get_random_test_tensor(size=(12, 5), is_float=True)
            input.requires_grad = True
            encr_input = AutogradCrypTensor(crypten.cryptensor(input))

            # perform forward-backward pass:
            output = getattr(input, func_name)(input).sum()
            encr_output = getattr(encr_input, func_name)(encr_input).sum()
            self._check(encr_output._tensor, output, "forward failed")
            self.assertTrue(encr_output.requires_grad, "requires_grad incorrect")
            output.backward()
            encr_output.backward()
            self._check(encr_input.grad, input.grad, "%s backward failed" % func_name)

    def test_autograd(self):
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
            encr_inputs = [AutogradCrypTensor(encr_input) for encr_input in encr_inputs]

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
                self._check(encr_inputs[idx].grad, inputs[idx].grad,
                            "backward failed")

    def test_autograd_repetition(self):
        """Tests running autograd on the same input repeatedly."""

        # create test case:
        input = get_random_test_tensor(size=(12, 5), is_float=True)
        input.requires_grad = True
        encr_input = AutogradCrypTensor(crypten.cryptensor(input))

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


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestAutograd.benchmarks_enabled = True
    unittest.main()
