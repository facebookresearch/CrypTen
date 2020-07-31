#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest
from test.multiprocess_test_case import (
    MultiProcessTestCase,
    get_random_linear,
    get_random_test_tensor,
    onehot,
)

import crypten
import crypten.communicator as comm
import torch
from crypten.common.rng import generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.encoder import FixedPointEncoder


class TestNN(object):
    """
        This class tests the crypten.nn package.
    """

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        if is_float_tensor(reference):
            diff = (tensor - reference).abs_()
            norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
            test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.2)
            test_passed = test_passed.gt(0).all().item() == 1
        else:
            test_passed = (tensor == reference).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result - Reference = %s" % (tensor - reference))
        self.assertTrue(test_passed, msg=msg)

    def _compute_reference_parameters(self, init_name, reference, model, learning_rate):
        for name, param in model.named_parameters(recurse=False):
            local_name = init_name + "_" + name
            reference[local_name] = (
                param.get_plain_text() - learning_rate * param.grad.get_plain_text()
            )
        for name, module in model._modules.items():
            local_name = init_name + "_" + name
            reference = self._compute_reference_parameters(
                local_name, reference, module, learning_rate
            )
        return reference

    def _check_reference_parameters(self, init_name, reference, model):
        for name, param in model.named_parameters(recurse=False):
            local_name = init_name + "_" + name
            self._check(param, reference[local_name], "parameter update failed")
        for name, module in model._modules.items():
            local_name = init_name + "_" + name
            self._check_reference_parameters(local_name, reference, module)

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communicator
        if self.rank >= 0:
            crypten.init()

    def test_from_shares(self):
        """Tests crypten.nn.Module.set_parameter_from_shares() functionality."""

        # create simple model:
        input_size, output_size = 3, 10
        model = crypten.nn.Linear(input_size, output_size)

        # helper function that creates arithmetically shared tensor of some size:
        def _generate_parameters(size):
            num_parties = int(self.world_size)
            reference = get_random_test_tensor(size=size, is_float=False)
            zero_shares = generate_random_ring_element((num_parties, *size))
            zero_shares = zero_shares - zero_shares.roll(1, dims=0)
            shares = list(zero_shares.unbind(0))
            shares[0] += reference
            return shares, reference

        # generate new set of parameters:
        all_shares, all_references = {}, {}
        for name, param in model.named_parameters():
            shares, reference = _generate_parameters(param.size())
            share = comm.get().scatter(shares, 0)
            all_shares[name] = share
            all_references[name] = reference

        # cannot load parameters from share when model is not encrypted:
        with self.assertRaises(AssertionError):
            for name, share in all_shares.items():
                model.set_parameter_from_shares(name, share)

        # cannot load shares into non-existent parameters:
        model.encrypt()
        with self.assertRaises(ValueError):
            model.set_parameter_from_shares("__DUMMY__", None)

        # load parameter shares into model and check results:
        for name, share in all_shares.items():
            model.set_parameter_from_shares(name, share)
        model.decrypt()
        encoder = FixedPointEncoder()
        for name, param in model.named_parameters():
            reference = encoder.decode(all_references[name])
            self.assertTrue(torch.allclose(param, reference))

    def test_global_avg_pool_module(self):
        """
        Tests the global average pool module with fixed 4-d test tensors
        """
        # construct basic input
        base_tensor = torch.Tensor([[2, 1], [3, 0]])
        all_init = []
        for i in range(-2, 3):
            all_init.append(torch.add(base_tensor, i))
        init_tensor = torch.stack(all_init, dim=2)
        init_tensor = init_tensor.unsqueeze(-1)
        reference = base_tensor.unsqueeze(-1).unsqueeze(-1)

        # create module
        encr_module = crypten.nn.GlobalAveragePool().encrypt()
        self.assertTrue(encr_module.encrypted, "module not encrypted")

        # check correctness for a variety of input sizes
        for i in range(1, 10):
            input = init_tensor.repeat(1, 1, i, i)
            encr_input = crypten.cryptensor(input)
            encr_output = encr_module(encr_input)
            self._check(encr_output, reference, "GlobalAveragePool failed")

    def test_dropout_module(self):
        """Tests the dropout module"""
        input_size = [3, 3, 3]
        prob_list = [0.2 * x for x in range(1, 5)]
        for module_name in ["Dropout", "Dropout2d", "Dropout3d"]:
            for prob in prob_list:
                for compute_gradients in [True, False]:
                    # generate inputs:
                    input = get_random_test_tensor(
                        size=input_size, is_float=True, ex_zero=True
                    )
                    input.requires_grad = True
                    encr_input = crypten.cryptensor(input)
                    encr_input.requires_grad = compute_gradients

                    # create PyTorch module:
                    module = getattr(torch.nn, module_name)(prob)
                    module.train()

                    # create encrypted CrypTen module:
                    encr_module = crypten.nn.from_pytorch(module, input)

                    # check that module properly encrypts / decrypts and
                    # check that encrypting with current mode properly
                    # performs no-op
                    for encrypted in [False, True, True, False, True]:
                        encr_module.encrypt(mode=encrypted)
                        if encrypted:
                            self.assertTrue(
                                encr_module.encrypted, "module not encrypted"
                            )
                        else:
                            self.assertFalse(encr_module.encrypted, "module encrypted")

                    # compare model outputs:
                    # compare the zero and non-zero entries of the encrypted tensor
                    # with a directly constructed plaintext tensor, since we cannot
                    # ensure that the randomization produces the same output
                    # for both encrypted and plaintext tensors
                    self.assertTrue(encr_module.training, "training value incorrect")
                    encr_output = encr_module(encr_input)
                    plaintext_output = encr_output.get_plain_text()
                    scaled_tensor = input / (1 - prob)
                    reference = plaintext_output.where(
                        plaintext_output == 0, scaled_tensor
                    )
                    self._check(encr_output, reference, "Dropout forward failed")

                    # check backward
                    # compare the zero and non-zero entries of the grad in
                    # the encrypted tensor with a directly constructed plaintext
                    # tensor: we do this because we cannot ensure that the
                    # randomization produces the same output for the input
                    # encrypted and plaintext tensors and so we cannot ensure
                    # that the grad in the input tensor is populated identically
                    all_ones = torch.ones(reference.size())
                    ref_grad = plaintext_output.where(plaintext_output == 0, all_ones)
                    ref_grad_input = ref_grad / (1 - prob)
                    encr_output.sum().backward()
                    if compute_gradients:
                        self._check(
                            encr_input.grad,
                            ref_grad_input,
                            "dropout backward on input failed",
                        )

                    # check testing mode for Dropout module
                    encr_module.train(mode=False)
                    encr_output = encr_module(encr_input)
                    result = encr_input.eq(encr_output)
                    result_plaintext = result.get_plain_text().bool()
                    self.assertTrue(
                        result_plaintext.all(), "dropout failed in test mode"
                    )

    def test_non_pytorch_modules(self):
        """
        Tests all non-container Modules in crypten.nn that do not have
        equivalent modules in PyTorch.
        """

        # input arguments for modules and input sizes:
        no_input_modules = ["Constant"]
        binary_modules = ["Add", "Sub", "Concat", "MatMul"]
        ex_zero_modules = []
        module_args = {
            "Add": (),
            "Concat": (0,),
            "Constant": (1.2,),
            "Exp": (),
            "Gather": (0,),
            "MatMul": (),
            "Mean": ([0], True),
            "Reshape": (),
            "Shape": (),
            "Sub": (),
            "Sum": ([0], True),
            "Squeeze": (0,),
            "Transpose": ([1, 3, 0, 2],),
            "Unsqueeze": (0,),
        }
        module_lambdas = {
            "Add": lambda x: x[0] + x[1],
            "Concat": lambda x: torch.cat((x[0], x[1])),
            "Constant": lambda _: torch.tensor(module_args["Constant"][0]),
            "Exp": lambda x: torch.exp(x),
            "Gather": lambda x: torch.from_numpy(
                x[0].numpy().take(x[1], module_args["Gather"][0])
            ),
            "MatMul": lambda x: torch.matmul(x[0], x[1]),
            "Mean": lambda x: torch.mean(
                x, dim=module_args["Mean"][0], keepdim=(module_args["Mean"][1] == 1)
            ),
            "Reshape": lambda x: x[0].reshape(x[1]),
            "Shape": lambda x: torch.tensor(x.size()).float(),
            "Sub": lambda x: x[0] - x[1],
            "Sum": lambda x: torch.sum(
                x, dim=module_args["Sum"][0], keepdim=(module_args["Sum"][1] == 1)
            ),
            "Squeeze": lambda x: x.squeeze(module_args["Squeeze"][0]),
            "Transpose": lambda x: torch.from_numpy(
                x.numpy().transpose(module_args["Transpose"][0])
            ),
            "Unsqueeze": lambda x: x.unsqueeze(module_args["Unsqueeze"][0]),
        }
        input_sizes = {
            "Add": (10, 12),
            "Concat": (2, 2),
            "Constant": (1,),
            "Exp": (10, 10, 10),
            "Gather": (4, 4, 4, 4),
            "MatMul": (4, 4),
            "Mean": (3, 3, 3),
            "Reshape": (1, 4),
            "Shape": (8, 3, 2),
            "Sub": (10, 12),
            "Sum": (3, 3, 3),
            "Squeeze": (1, 12, 6),
            "Transpose": (1, 2, 3, 4),
            "Unsqueeze": (8, 3),
        }
        additional_inputs = {
            "Gather": torch.tensor([[1, 2], [0, 3]]),
            "Reshape": torch.Size([2, 2]),
        }
        module_attributes = {
            # each attribute has two parameters: the name, and a bool indicating
            # whether the value should be wrapped into a list when the module is created
            "Add": [],
            "Exp": [],
            "Concat": [("axis", False)],
            "Constant": [("value", False)],
            "Gather": [("axis", False)],
            "MatMul": [],
            "Mean": [("axes", False), ("keepdims", False)],
            "Reshape": [],
            "Shape": [],
            "Sub": [],
            "Sum": [("axes", False), ("keepdims", False)],
            "Squeeze": [("axes", True)],
            "Transpose": [("perm", False)],
            "Unsqueeze": [("axes", True)],
        }
        # loop over all modules:
        for module_name in module_args.keys():
            # create encrypted CrypTen module:
            encr_module = getattr(crypten.nn, module_name)(*module_args[module_name])
            encr_module.encrypt()
            self.assertTrue(encr_module.encrypted, "module not encrypted")

            # generate inputs:
            inputs, encr_inputs = None, None
            ex_zero_values = module_name in ex_zero_modules
            if module_name in binary_modules:
                inputs = [
                    get_random_test_tensor(
                        size=input_sizes[module_name],
                        is_float=True,
                        ex_zero=ex_zero_values,
                    )
                    for _ in range(2)
                ]
                encr_inputs = [crypten.cryptensor(input) for input in inputs]
            elif module_name not in no_input_modules:
                inputs = get_random_test_tensor(
                    size=input_sizes[module_name], is_float=True, ex_zero=ex_zero_values
                )
                encr_inputs = crypten.cryptensor(inputs)

            # some modules take additonal indices as input:
            if module_name in additional_inputs:
                if not isinstance(inputs, (list, tuple)):
                    inputs, encr_inputs = [inputs], [encr_inputs]
                inputs.append(additional_inputs[module_name])
                # encrypt only torch tensor inputs, not shapes
                if torch.is_tensor(inputs[-1]):
                    encr_inputs.append(crypten.cryptensor(inputs[-1]))
                else:
                    encr_inputs.append(inputs[-1])

            # compare model outputs:
            reference = module_lambdas[module_name](inputs)
            encr_output = encr_module(encr_inputs)
            self._check(encr_output, reference, "%s failed" % module_name)

            # create attributes for static from_onnx function
            local_attr = {}
            for i, attr_tuple in enumerate(module_attributes[module_name]):
                attr_name, wrap_attr_in_list = attr_tuple
                if wrap_attr_in_list:
                    local_attr[attr_name] = [module_args[module_name][i]]
                else:
                    local_attr[attr_name] = module_args[module_name][i]

            # Update ReduceSum/ReduceMean module attributes, since the module and
            # from_onnx path are different
            if module_name == "ReduceSum":
                local_attr["keepdims"] = 1 if module_args["ReduceSum"][1] is True else 0
            if module_name == "ReduceMean":
                local_attr["keepdims"] = (
                    1 if module_args["ReduceMean"][1] is True else 0
                )

            # compare model outputs using the from_onnx static function
            module = getattr(crypten.nn, module_name).from_onnx(attributes=local_attr)
            encr_module_onnx = module.encrypt()
            encr_output = encr_module_onnx(encr_inputs)
            self._check(encr_output, reference, "%s failed" % module_name)

    def test_pytorch_modules(self):
        """
        Tests all non-container Modules in crypten.nn that have equivalent
        modules in PyTorch.
        """

        # input arguments for modules and input sizes:
        module_args = {
            "AdaptiveAvgPool2d": (2,),
            "AvgPool2d": (2,),
            # "BatchNorm1d": (400,),  # FIXME: Unit tests claim gradients are incorrect.
            # "BatchNorm2d": (3,),
            # "BatchNorm3d": (6,),
            "ConstantPad1d": (3, 1.0),
            "ConstantPad2d": (2, 2.0),
            "ConstantPad3d": (1, 0.0),
            "Conv1d": (3, 6, 5),
            "Conv2d": (3, 6, 5),
            "Linear": (400, 120),
            "MaxPool2d": (2,),
            "ReLU": (),
            "Sigmoid": (),
            "Softmax": (0,),
            "LogSoftmax": (0,),
        }
        input_sizes = {
            "AdaptiveAvgPool2d": (1, 3, 32, 32),
            "AvgPool2d": (1, 3, 32, 32),
            "BatchNorm1d": (8, 400),
            "BatchNorm2d": (8, 3, 32, 32),
            "BatchNorm3d": (8, 6, 32, 32, 4),
            "ConstantPad1d": (9,),
            "ConstantPad2d": (3, 6),
            "ConstantPad3d": (4, 2, 7),
            "Conv1d": (1, 3, 32),
            "Conv2d": (1, 3, 32, 32),
            "Linear": (1, 400),
            "MaxPool2d": (1, 2, 32, 32),
            "ReLU": (1, 3, 32, 32),
            "Sigmoid": (8, 3, 32, 32),
            "Softmax": (5, 5, 5),
            "LogSoftmax": (5, 5, 5),
        }

        # loop over all modules:
        for module_name in module_args.keys():
            for compute_gradients in [True, False]:

                # generate inputs:
                input = get_random_test_tensor(
                    size=input_sizes[module_name], is_float=True
                )
                input.requires_grad = True
                encr_input = crypten.cryptensor(input)
                encr_input.requires_grad = compute_gradients

                # create PyTorch module:
                module = getattr(torch.nn, module_name)(*module_args[module_name])
                module.train()

                # create encrypted CrypTen module:
                encr_module = crypten.nn.from_pytorch(module, input)

                # check that module properly encrypts / decrypts and
                # check that encrypting with current mode properly performs no-op
                for encrypted in [False, True, True, False, True]:
                    encr_module.encrypt(mode=encrypted)
                    if encrypted:
                        self.assertTrue(encr_module.encrypted, "module not encrypted")
                    else:
                        self.assertFalse(encr_module.encrypted, "module encrypted")
                    for key in ["weight", "bias"]:
                        if hasattr(module, key):  # if PyTorch model has key
                            encr_param = None

                            # find that key in the crypten.nn.Graph:
                            if isinstance(encr_module, crypten.nn.Graph):
                                for encr_node in encr_module.modules():
                                    if hasattr(encr_node, key):
                                        encr_param = getattr(encr_node, key)
                                        break

                            # or get it from the crypten Module directly:
                            else:
                                encr_param = getattr(encr_module, key)

                            # compare with reference:
                            # NOTE: Because some parameters are initialized randomly
                            # with different values on each process, we only want to
                            # check that they are consistent with source parameter value
                            reference = getattr(module, key)
                            src_reference = comm.get().broadcast(reference, src=0)
                            msg = "parameter %s in %s incorrect" % (key, module_name)
                            if not encrypted:
                                encr_param = crypten.cryptensor(encr_param)
                            self._check(encr_param, src_reference, msg)

                # compare model outputs:
                self.assertTrue(encr_module.training, "training value incorrect")
                reference = module(input)
                encr_output = encr_module(encr_input)
                self._check(encr_output, reference, "%s forward failed" % module_name)

                # test backward pass:
                reference.sum().backward()
                encr_output.sum().backward()
                if compute_gradients:
                    self._check(
                        encr_input.grad,
                        input.grad,
                        "%s backward on input failed" % module_name,
                    )
                else:
                    self.assertIsNone(encr_input.grad)
                for name, param in module.named_parameters():
                    encr_param = getattr(encr_module, name)
                    self._check(
                        encr_param.grad,
                        param.grad,
                        "%s backward on %s failed" % (module_name, name),
                    )

    def test_linear(self):
        """
        Tests crypten.nn.Linear module.
        """
        dims = [(40, 80), (80, 1), (10, 10)]
        sizes = [(1, 40), (4, 80), (6, 4, 2, 10)]
        for compute_gradients in [True, False]:
            for dim, size in zip(dims, sizes):
                # generate inputs:
                input = get_random_test_tensor(size=size, is_float=True)
                input.requires_grad = True
                encr_input = crypten.cryptensor(input)
                encr_input.requires_grad = compute_gradients

                # create PyTorch module:
                module = torch.nn.Linear(*dim)
                module.train()

                # create encrypted CrypTen module:
                encr_module = crypten.nn.Linear(*dim)
                for n, p in module.named_parameters():
                    p = comm.get().broadcast(p, src=0)
                    encr_module.set_parameter(n, p)
                encr_module.encrypt().train()

                # compare model outputs:
                self.assertTrue(encr_module.training, "training value incorrect")
                reference = module(input)
                encr_output = encr_module(encr_input)
                self._check(encr_output, reference, "Linear forward failed")

                # test backward pass:
                reference.backward(torch.ones(reference.size()))
                encr_output.backward(encr_output.new(torch.ones(encr_output.size())))
                if compute_gradients:
                    self._check(
                        encr_input.grad, input.grad, "Linear backward on input failed"
                    )
                else:
                    self.assertIsNone(encr_input.grad)
                for name, param in module.named_parameters():
                    encr_param = getattr(encr_module, name)
                    self._check(
                        encr_param.grad,
                        param.grad,
                        "Linear backward on %s failed" % name,
                    )

    def test_sequential(self):
        """
        Tests crypten.nn.Sequential module.
        """

        # try networks of different depth:
        for num_layers in range(1, 6):
            for compute_gradients in [True, False]:

                # construct sequential container:
                input_size = (3, 10)
                output_size = (input_size[0], input_size[1] - num_layers)
                layer_idx = range(input_size[1], output_size[1], -1)
                module_list = [
                    crypten.nn.Linear(num_feat, num_feat - 1) for num_feat in layer_idx
                ]
                sequential = crypten.nn.Sequential(*module_list)
                sequential.encrypt()

                # check container:
                self.assertTrue(sequential.encrypted, "nn.Sequential not encrypted")
                for module in sequential.modules():
                    self.assertTrue(module.encrypted, "module not encrypted")
                assert sum(1 for _ in sequential.modules()) == len(
                    module_list
                ), "nn.Sequential contains incorrect number of modules"

                # construct test input and run through sequential container:
                input = get_random_test_tensor(size=input_size, is_float=True)
                encr_input = crypten.cryptensor(input)
                encr_input.requires_grad = compute_gradients
                encr_output = sequential(encr_input)

                # compute reference output:
                encr_reference = encr_input
                for module in sequential.modules():
                    encr_reference = module(encr_reference)
                reference = encr_reference.get_plain_text()

                # compare output to reference:
                self._check(encr_output, reference, "nn.Sequential forward failed")

    def test_graph(self):
        """
        Tests crypten.nn.Graph module.
        """
        for compute_gradients in [True, False]:

            # define test case:
            input_size = (3, 10)
            input = get_random_test_tensor(size=input_size, is_float=True)
            encr_input = crypten.cryptensor(input)
            encr_input.requires_grad = compute_gradients

            # test residual block with subsequent linear layer:
            graph = crypten.nn.Graph("input", "output")
            linear1 = get_random_linear(input_size[1], input_size[1])
            linear2 = get_random_linear(input_size[1], input_size[1])
            graph.add_module(
                "linear", crypten.nn.from_pytorch(linear1, input), ["input"]
            )
            graph.add_module("residual", crypten.nn.Add(), ["input", "linear"])
            graph.add_module(
                "output", crypten.nn.from_pytorch(linear2, input), ["residual"]
            )
            graph.encrypt()

            # check container:
            self.assertTrue(graph.encrypted, "nn.Graph not encrypted")
            for module in graph.modules():
                self.assertTrue(module.encrypted, "module not encrypted")
            assert (
                sum(1 for _ in graph.modules()) == 3
            ), "nn.Graph contains incorrect number of modules"

            # compare output to reference:
            encr_output = graph(encr_input)
            reference = linear2(linear1(input) + input)
            self._check(encr_output, reference, "nn.Graph forward failed")

    def test_losses(self):
        """
        Tests all Losses implemented in crypten.nn.
        """

        # create test tensor:
        input = get_random_test_tensor(max_value=0.999, is_float=True).abs() + 0.001
        target = get_random_test_tensor(max_value=0.999, is_float=True).abs() + 0.001
        encrypted_input = crypten.cryptensor(input)
        encrypted_target = crypten.cryptensor(target)

        # test forward() function of all simple losses:
        for loss_name in ["BCELoss", "BCEWithLogitsLoss", "L1Loss", "MSELoss"]:
            for skip_forward in [False, True]:
                enc_loss_object = getattr(torch.nn, loss_name)()
                self.assertEqual(
                    enc_loss_object.reduction, "mean", "Reduction used is not 'mean'"
                )

                input.requires_grad = True
                input.grad = None
                loss = getattr(torch.nn, loss_name)()(input, target)
                if not skip_forward:
                    encrypted_loss = getattr(crypten.nn, loss_name)()(
                        encrypted_input, encrypted_target
                    )
                    self._check(encrypted_loss, loss, "%s failed" % loss_name)

                encrypted_input.requires_grad = True
                encrypted_input.grad = None
                encrypted_loss = getattr(crypten.nn, loss_name)(
                    skip_forward=skip_forward
                )(encrypted_input, encrypted_target)
                if not skip_forward:
                    self._check(encrypted_loss, loss, "%s failed" % loss_name)

                # Check backward
                loss.backward()
                encrypted_loss.backward()
                self._check(
                    encrypted_input.grad, input.grad, "%s grad failed" % loss_name
                )

        # test forward() function of cross-entropy loss:
        batch_size, num_targets = 16, 5
        input = get_random_test_tensor(size=(batch_size, num_targets), is_float=True)
        target = get_random_test_tensor(
            size=(batch_size,), max_value=num_targets - 1
        ).abs()
        encrypted_input = crypten.cryptensor(input)
        encrypted_target = crypten.cryptensor(onehot(target, num_targets=num_targets))
        enc_loss_object = crypten.nn.CrossEntropyLoss()
        self.assertEqual(
            enc_loss_object.reduction, "mean", "Reduction used is not 'mean'"
        )

        loss = torch.nn.CrossEntropyLoss()(input, target)
        encrypted_loss = crypten.nn.CrossEntropyLoss()(
            encrypted_input, encrypted_target
        )
        self._check(encrypted_loss, loss, "cross-entropy loss failed")
        encrypted_input.requires_grad = True
        encrypted_target.requires_grad = True
        encrypted_loss = crypten.nn.CrossEntropyLoss()(
            encrypted_input, encrypted_target
        )
        self._check(encrypted_loss, loss, "cross-entropy loss failed")

    def test_getattr_setattr(self):
        """Tests the __getattr__ and __setattr__ functions"""

        tensor1 = get_random_test_tensor(size=(3, 3), is_float=True)
        tensor2 = get_random_test_tensor(size=(3, 3), is_float=True)

        class ExampleNet(crypten.nn.Module):
            def __init__(self):
                super(ExampleNet, self).__init__()
                self.fc1 = crypten.nn.Linear(20, 1)
                sample_buffer = tensor1
                self.register_buffer("sample_buffer", sample_buffer)
                sample_param = tensor2
                self.register_parameter("sample_param", sample_param)

            def forward(self, x):
                out = self.fc1(x)
                return out

        model = ExampleNet()
        model.encrypt()

        self.assertTrue("fc1" in model._modules.keys(), "modules __setattr__ failed")
        self._check(model.sample_buffer, tensor1, "buffer __getattr__ failed")
        self._check(model.sample_param, tensor2, "parameter __getattr__ failed")
        self.assertTrue(
            isinstance(model.fc1, crypten.nn.Linear), "modules __getattr__ failed"
        )

        """
        assign to model.weight should change model._parameters["weight"]
        """
        model.fc1.weight = torch.nn.Parameter(torch.zeros((2, 3)))

        self.assertEqual(
            model.fc1._parameters["weight"].tolist(),
            torch.nn.Parameter(torch.zeros((2, 3))).tolist(),
        )

        """
        assign to  model._parameters["weight"] should change model.weight
        """
        model.fc1._parameters["weight"] = torch.nn.Parameter(torch.ones((2, 3)))
        self.assertEqual(
            model.fc1.weight.tolist(), torch.nn.Parameter(torch.ones((2, 3))).tolist()
        )

        """
        assign to  model._buffers["bufferedItem"] should change model.bufferedItem
        """
        model.fc1._buffers["bufferedItem"] = torch.nn.Parameter(torch.ones((2, 3)))
        self.assertEqual(
            model.fc1.bufferedItem.tolist(),
            torch.nn.Parameter(torch.ones((2, 3))).tolist(),
        )

        """
        assign to model.weight should change model._parameters["weight"]
        """
        model.fc1.bufferedItem = torch.nn.Parameter(torch.zeros((2, 3)))

        self.assertEqual(
            model.fc1._buffers["bufferedItem"].tolist(),
            torch.nn.Parameter(torch.zeros((2, 3))).tolist(),
        )

    def test_training(self):
        """
        Tests training of simple model in crypten.nn.
        """

        # create MLP with one hidden layer:
        learning_rate = 0.1
        batch_size, num_inputs, num_intermediate, num_outputs = 8, 10, 5, 1
        model = crypten.nn.Sequential(
            crypten.nn.Linear(num_inputs, num_intermediate),
            crypten.nn.ReLU(),
            crypten.nn.Linear(num_intermediate, num_outputs),
        )
        model.train()
        model.encrypt()
        loss = crypten.nn.MSELoss()

        # perform training iterations:
        for _ in range(10):
            for compute_gradients in [True, False]:

                # get training sample:
                input = get_random_test_tensor(
                    size=(batch_size, num_inputs), is_float=True
                )
                target = input.mean(dim=1, keepdim=True)

                # encrypt training sample:
                input = crypten.cryptensor(input)
                target = crypten.cryptensor(target)
                if compute_gradients:
                    input.requires_grad = True
                    target.requires_grad = True

                # perform forward pass:
                output = model(input)
                loss_value = loss(output, target)

                # set gradients to "zero" (setting to None is more efficient):
                model.zero_grad()
                for param in model.parameters():
                    self.assertIsNone(param.grad, "zero_grad did not reset gradients")

                # perform backward pass:
                loss_value.backward()

                # perform parameter update:
                reference = {}
                reference = self._compute_reference_parameters(
                    "", reference, model, learning_rate
                )
                model.update_parameters(learning_rate)
                self._check_reference_parameters("", reference, model)

    def test_custom_module_training(self):
        """Tests training CrypTen models created directly using the crypten.nn.Module"""
        BATCH_SIZE = 32
        NUM_FEATURES = 3

        class ExampleNet(crypten.nn.Module):
            def __init__(self):
                super(ExampleNet, self).__init__()
                self.fc1 = crypten.nn.Linear(NUM_FEATURES, BATCH_SIZE)
                self.fc2 = crypten.nn.Linear(BATCH_SIZE, 2)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        model = ExampleNet()

        x_orig = get_random_test_tensor(size=(BATCH_SIZE, NUM_FEATURES), is_float=True)
        # y is a linear combo of x to ensure network can easily learn pattern
        y_orig = (2 * x_orig.mean(dim=1)).gt(0).long()
        y_one_hot = onehot(y_orig, num_targets=2)

        # encrypt training sample:
        x_train = crypten.cryptensor(x_orig, requires_grad=True)
        y_train = crypten.cryptensor(y_one_hot)

        for loss_name in ["BCELoss", "CrossEntropyLoss", "MSELoss"]:
            # create loss function
            loss = getattr(crypten.nn, loss_name)()

            # create encrypted model
            model.train()
            model.encrypt()

            num_epochs = 3
            learning_rate = 0.001

            for i in range(num_epochs):
                output = model(x_train)
                if loss_name == "MSELoss":
                    output_norm = output
                else:
                    output_norm = output.softmax(1)
                loss_value = loss(output_norm, y_train)

                # set gradients to "zero"
                model.zero_grad()
                for param in model.parameters():
                    self.assertIsNone(param.grad, "zero_grad did not reset gradients")

                # perform backward pass:
                loss_value.backward()
                for param in model.parameters():
                    if param.requires_grad:
                        self.assertIsNotNone(
                            param.grad, "required parameter gradient not created"
                        )

                # update parameters
                orig_parameters, upd_parameters = {}, {}
                orig_parameters = self._compute_reference_parameters(
                    "", orig_parameters, model, 0
                )
                model.update_parameters(learning_rate)
                upd_parameters = self._compute_reference_parameters(
                    "", upd_parameters, model, learning_rate
                )

                parameter_changed = False
                for name, value in orig_parameters.items():
                    if param.requires_grad and param.grad is not None:
                        unchanged = torch.allclose(upd_parameters[name], value)
                        if unchanged is False:
                            parameter_changed = True
                        self.assertTrue(
                            parameter_changed, "no parameter changed in training step"
                        )

                # record initial and current loss
                if i == 0:
                    orig_loss = loss_value.get_plain_text()
                curr_loss = loss_value.get_plain_text()

            # check that the loss has decreased after training
            self.assertTrue(
                curr_loss.item() < orig_loss.item(),
                "loss has not decreased after training",
            )

    def test_batchnorm_module(self):
        """Test module correctly sets and updates running stats"""
        batchnorm_fn_and_size = (
            ("BatchNorm1d", (500, 10, 3)),
            ("BatchNorm2d", (600, 7, 4, 20)),
            ("BatchNorm3d", (800, 5, 4, 8, 15)),
        )
        for batchnorm_fn, size in batchnorm_fn_and_size:
            for is_trainning in (True, False):
                tensor = get_random_test_tensor(size=size, is_float=True)
                tensor.requires_grad = True
                encrypted_input = crypten.cryptensor(tensor, requires_grad=True)

                C = size[1]
                weight = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                bias = get_random_test_tensor(size=[C], max_value=1, is_float=True)
                weight.requires_grad = True
                bias.requires_grad = True

                # dimensions for mean and variance
                stats_dimensions = list(range(tensor.dim()))
                # perform on C dimension for tensor of shape (N, C, +)
                stats_dimensions.pop(1)

                # check running stats initial
                enc_model = getattr(crypten.nn.module, batchnorm_fn)(C).encrypt()
                plain_model = getattr(torch.nn.modules, batchnorm_fn)(C)
                stats = ["running_var", "running_mean"]
                for stat in stats:
                    self._check(
                        enc_model._buffers[stat],
                        plain_model._buffers[stat],
                        f"{stat} initial module value incorrect",
                    )

                # set trainning mode
                plain_model.training = is_trainning
                enc_model.training = is_trainning

                # check running_stats update
                enc_model.forward(encrypted_input)
                plain_model.forward(tensor)
                for stat in stats:
                    self._check(
                        enc_model._buffers[stat],
                        plain_model._buffers[stat],
                        f"{stat} momentum update in module incorrect",
                    )

    def test_unencrypted_modules(self):
        """Tests crypten.Modules without encrypting them."""

        # generate input:
        input_size = (32, 16)
        output_size = (input_size[0], 8)
        sample = get_random_test_tensor(size=input_size, is_float=True)
        target = get_random_test_tensor(size=output_size, is_float=True)

        # create model and criterion:
        linear = crypten.nn.Linear(input_size[1], output_size[1])
        criterion = crypten.nn.MSELoss()

        # function running the actual test:
        def _run_test(_sample, _target):

            # forward pass fails when feeding encrypted input into unencrypted model:
            linear.zero_grad()
            if not linear.encrypted and not torch.is_tensor(_sample):
                with self.assertRaises(RuntimeError):
                    output = linear(_sample)
                return

            # forward pass succeeds in other cases:
            output = linear(_sample)
            loss = criterion(output, _target)
            self.assertIsNotNone(loss)

            # backward pass succeeds in other cases:
            loss.backward()
            for param in linear.parameters():
                self.assertIsNotNone(param.grad)

            # test parameter update:
            original_params = [param.clone() for param in linear.parameters()]
            linear.update_parameters(1.0)
            for idx, param in enumerate(linear.parameters()):
                diff = param.sub(original_params[idx]).abs().mean()
                if isinstance(diff, crypten.CrypTensor):
                    diff = diff.get_plain_text()
                self.assertGreater(diff.item(), 1e-4)

        # test both tensor types in models with and without encryption:
        for encrypted in [False, True, False, True]:
            linear.encrypt(mode=encrypted)
            _run_test(sample, target)
            _run_test(crypten.cryptensor(sample), crypten.cryptensor(target))

    def test_state_dict(self):
        """
        Tests dumping and loading of state dicts.
        """

        def _check_equal(t1, t2):
            """
            Checks whether to tensors are identical.
            """
            if isinstance(t1, torch.nn.parameter.Parameter):
                t1 = t1.data
            if isinstance(t2, torch.nn.parameter.Parameter):
                t2 = t2.data
            self.assertEqual(type(t1), type(t2))
            if isinstance(t1, crypten.CrypTensor):
                t1 = t1.get_plain_text()
                t2 = t2.get_plain_text()
            self.assertTrue(t1.eq(t2).all())

        def _check_state_dict(model, state_dict):
            """
            Checks if state_dict matches parameters in model.
            """
            print(model)

            # get all parameters, buffers, and names from model:
            params_buffers = {}
            for func in ["named_parameters", "named_buffers"]:
                params_buffers.update({k: v for k, v in getattr(model, func)()})

            # do all the checks:
            self.assertEqual(len(params_buffers), len(state_dict))
            for name, param_or_buffer in params_buffers.items():
                self.assertIn(name, state_dict)
                _check_equal(state_dict[name], param_or_buffer)

        # test for individual modules:
        module_args = {
            "BatchNorm1d": (400,),
            "BatchNorm2d": (3,),
            "BatchNorm3d": (6,),
            "Conv1d": (3, 6, 5),
            "Conv2d": (3, 6, 5),
            "Linear": (400, 120),
        }
        for module_name, args in module_args.items():
            for encrypt in [False, True]:

                # create module and get state dict:
                module = getattr(crypten.nn, module_name)(*args)
                if encrypt:
                    module.encrypt()
                state_dict = module.state_dict()
                _check_state_dict(module, state_dict)

                # load state dict into fresh module:
                new_module = getattr(crypten.nn, module_name)(*args)
                if encrypt:
                    with self.assertRaises(AssertionError):
                        new_module.load_state_dict(state_dict)
                    new_module.encrypt()
                new_module.load_state_dict(state_dict)
                _check_state_dict(new_module, state_dict)

        # tests for model that is sequence of modules:
        for num_layers in range(1, 6):
            for encrypt in [False, True]:

                # some variables that we need:
                input_size = (3, 10)
                output_size = (input_size[0], input_size[1] - num_layers)
                layer_idx = range(input_size[1], output_size[1], -1)

                # construct sequential model:
                module_list = [
                    crypten.nn.Linear(num_feat, num_feat - 1) for num_feat in layer_idx
                ]
                model = crypten.nn.Sequential(*module_list)
                if encrypt:
                    model.encrypt()

                # check state dict:
                state_dict = model.state_dict()
                _check_state_dict(model, state_dict)

                # load state dict into fresh model:
                state_dict = model.state_dict()
                module_list = [
                    crypten.nn.Linear(num_feat, num_feat - 1) for num_feat in layer_idx
                ]
                new_model = crypten.nn.Sequential(*module_list)
                if encrypt:
                    with self.assertRaises(AssertionError):
                        new_model.load_state_dict(state_dict)
                    new_model.encrypt()
                new_model.load_state_dict(state_dict)

                # check new model:
                _check_state_dict(model, state_dict)

    def test_to(self):
        """Test Module.to, Module.cpu, and Module.cuda"""
        module_list = [crypten.nn.Linear(10, 10) for _ in range(3)]
        model = crypten.nn.Sequential(*module_list)

        model_cpu = model.to("cpu")
        cpu = torch.device("cpu")
        for param in model_cpu.parameters():
            self.assertEqual(param.device, cpu)
        for buffer in model_cpu.buffers():
            self.assertEqual(buffer.device, cpu)

        model_cpu = model.cpu()
        for param in model_cpu.parameters():
            self.assertEqual(param.device, cpu)
        for buffer in model_cpu.buffers():
            self.assertEqual(buffer.device, cpu)

        if torch.cuda.is_available():
            cuda = torch.device("cuda:0")
            model_cuda = model.cuda()
            for param in model_cuda.parameters():
                self.assertEqual(param.device, cuda)
            for buffer in model_cuda.buffers():
                self.assertEqual(buffer.device, cuda)

            model_cuda = model.to("cuda:0")
            for param in model_cuda.parameters():
                self.assertEqual(param.device, cuda)
            for buffer in model_cuda.buffers():
                self.assertEqual(buffer.device, cuda)


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestNN):
    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedFirstParty)
        super(TestTFP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestNN):
    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)
        super(TestTTP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
