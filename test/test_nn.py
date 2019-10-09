#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
from crypten.autograd_cryptensor import AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor


class TestNN(MultiProcessTestCase):
    """
        This class tests the crypten.nn package.
    """

    benchmarks_enabled = False

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
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
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

    def test_non_pytorch_modules(self):
        """
        Tests all non-container Modules in crypten.nn that do not have
        equivalent modules in PyTorch.
        """

        # input arguments for modules and input sizes:
        no_input_modules = ["Constant"]
        binary_modules = ["Add", "Sub", "Concat"]
        module_args = {
            "Add": (),
            "Concat": (0,),
            "Constant": (1.2,),
            "Gather": (0,),
            "Reshape": (),
            "Shape": (),
            "Sub": (),
            "Squeeze": (0,),
            "Unsqueeze": (0,),
        }
        module_lambdas = {
            "Add": lambda x: x[0] + x[1],
            "Concat": lambda x: torch.cat((x[0], x[1])),
            "Constant": lambda _: torch.tensor(module_args["Constant"][0]),
            "Gather": lambda x: torch.from_numpy(
                x[0].numpy().take(x[1], module_args["Gather"][0])
            ),
            "Reshape": lambda x: x[0].reshape(x[1].tolist()),
            "Shape": lambda x: torch.tensor(x.size()).float(),
            "Sub": lambda x: x[0] - x[1],
            "Squeeze": lambda x: x.squeeze(module_args["Squeeze"][0]),
            "Unsqueeze": lambda x: x.unsqueeze(module_args["Unsqueeze"][0]),
        }
        input_sizes = {
            "Add": (10, 12),
            "Concat": (2, 2),
            "Constant": (1,),
            "Gather": (4, 4, 4, 4),
            "Reshape": (1, 4),
            "Shape": (8, 3, 2),
            "Sub": (10, 12),
            "Squeeze": (1, 12, 6),
            "Unsqueeze": (8, 3),
        }
        additional_inputs = {
            "Gather": torch.tensor([[1, 2], [0, 3]]),
            "Reshape": torch.tensor([2, 2]),
        }
        module_attributes = {
            "Add": [],
            "Concat": [("axis", int)],
            "Constant": [("value", int)],
            "Gather": [("axis", int)],
            "Reshape": [],
            "Shape": [],
            "Sub": [],
            "Squeeze": [("axes", list)],
            "Unsqueeze": [("axes", list)],
        }
        # loop over all modules:
        for module_name in module_args.keys():

            # create encrypted CrypTen module:
            encr_module = getattr(crypten.nn, module_name)(*module_args[module_name])
            encr_module.encrypt()
            self.assertTrue(encr_module.encrypted, "module not encrypted")

            # generate inputs:
            inputs, encr_inputs = None, None
            if module_name in binary_modules:
                inputs = [
                    get_random_test_tensor(size=input_sizes[module_name], is_float=True)
                    for _ in range(2)
                ]
                encr_inputs = [crypten.cryptensor(input) for input in inputs]
            elif module_name not in no_input_modules:
                inputs = get_random_test_tensor(
                    size=input_sizes[module_name], is_float=True
                )
                encr_inputs = crypten.cryptensor(inputs)

            # some modules take additonal indices as input:
            if module_name in additional_inputs:
                if not isinstance(inputs, (list, tuple)):
                    inputs, encr_inputs = [inputs], [encr_inputs]
                inputs.append(additional_inputs[module_name])
                encr_inputs.append(crypten.cryptensor(inputs[-1]))

            # compare model outputs:
            reference = module_lambdas[module_name](inputs)
            encr_output = encr_module(encr_inputs)
            self._check(encr_output, reference, "%s failed" % module_name)

            # create attributes for static from_onnx function
            local_attr = {}
            for i, attr_tuple in enumerate(module_attributes[module_name]):
                attr_name, attr_type = attr_tuple
                if attr_type == list:
                    local_attr[attr_name] = [module_args[module_name][i]]
                else:
                    local_attr[attr_name] = module_args[module_name][i]

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
            "Conv2d": (3, 6, 5),
            "Linear": (400, 120),
            "MaxPool2d": (2,),
            "ReLU": (),
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
            "Conv2d": (1, 3, 32, 32),
            "Linear": (1, 400),
            "MaxPool2d": (1, 2, 32, 32),
            "ReLU": (1, 3, 32, 32),
        }

        # loop over all modules:
        for module_name in module_args.keys():
            for wrap in [True, False]:

                # generate inputs:
                input = get_random_test_tensor(
                    size=input_sizes[module_name], is_float=True
                )
                input.requires_grad = True
                encr_input = crypten.cryptensor(input)
                if wrap:
                    encr_input = AutogradCrypTensor(encr_input)

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
                reference.backward(torch.ones(reference.size()))
                encr_output.backward()
                if wrap:  # you cannot get input gradients on MPCTensor inputs
                    self._check(
                        encr_input.grad,
                        input.grad,
                        "%s backward on input failed" % module_name,
                    )
                else:
                    self.assertFalse(hasattr(encr_input, "grad"))
                for name, param in module.named_parameters():
                    encr_param = getattr(encr_module, name)
                    self._check(
                        encr_param.grad,
                        param.grad,
                        "%s backward on %s failed" % (module_name, name),
                    )

    def test_sequential(self):
        """
        Tests crypten.nn.Sequential module.
        """

        # try networks of different depth:
        for num_layers in range(1, 6):
            for wrap in [True, False]:

                # construct sequential container:
                input_size = (3, 10)
                output_size = (input_size[0], input_size[1] - num_layers)
                layer_idx = range(input_size[1], output_size[1], -1)
                module_list = [
                    crypten.nn.Linear(num_feat, num_feat - 1) for num_feat in layer_idx
                ]
                sequential = crypten.nn.Sequential(module_list)
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
                if wrap:
                    encr_input = AutogradCrypTensor(encr_input)
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
        for wrap in [True, False]:

            # define test case:
            input_size = (3, 10)
            input = get_random_test_tensor(size=input_size, is_float=True)
            encr_input = crypten.cryptensor(input)
            if wrap:
                encr_input = AutogradCrypTensor(encr_input)

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
        for loss_name in ["BCELoss", "L1Loss", "MSELoss"]:
            loss = getattr(torch.nn, loss_name)()(input, target)
            encrypted_loss = getattr(crypten.nn, loss_name)()(
                encrypted_input, encrypted_target
            )
            self._check(encrypted_loss, loss, "%s failed" % loss_name)
            encrypted_loss = getattr(crypten.nn, loss_name)()(
                AutogradCrypTensor(encrypted_input),
                AutogradCrypTensor(encrypted_target),
            )
            self._check(encrypted_loss, loss, "%s failed" % loss_name)

        # test forward() function of cross-entropy loss:
        batch_size, num_targets = 16, 5
        input = get_random_test_tensor(size=(batch_size, num_targets), is_float=True)
        target = get_random_test_tensor(
            size=(batch_size,), max_value=num_targets - 1
        ).abs()
        encrypted_input = crypten.cryptensor(input)
        encrypted_target = crypten.cryptensor(onehot(target, num_targets=num_targets))
        loss = torch.nn.CrossEntropyLoss()(input, target)
        encrypted_loss = crypten.nn.CrossEntropyLoss()(
            encrypted_input, encrypted_target
        )
        self._check(encrypted_loss, loss, "cross-entropy loss failed")
        encrypted_loss = crypten.nn.CrossEntropyLoss()(
            AutogradCrypTensor(encrypted_input), AutogradCrypTensor(encrypted_target)
        )
        self._check(encrypted_loss, loss, "cross-entropy loss failed")

    def test_training(self):
        """
        Tests training of simple model in crypten.nn.
        """

        # create MLP with one hidden layer:
        learning_rate = 0.1
        batch_size, num_inputs, num_intermediate, num_outputs = 8, 10, 5, 1
        model = crypten.nn.Sequential(
            [
                crypten.nn.Linear(num_inputs, num_intermediate),
                crypten.nn.ReLU(),
                crypten.nn.Linear(num_intermediate, num_outputs),
            ]
        )
        model.train()
        model.encrypt()
        loss = crypten.nn.MSELoss()

        # perform training iterations:
        for _ in range(10):
            for wrap in [True, False]:

                # get training sample:
                input = get_random_test_tensor(
                    size=(batch_size, num_inputs), is_float=True
                )
                target = input.mean(dim=1, keepdim=True)

                # encrypt training sample:
                input = crypten.cryptensor(input)
                target = crypten.cryptensor(target)
                if wrap:
                    input = AutogradCrypTensor(input)
                    target = AutogradCrypTensor(target)

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

    def test_from_pytorch_training(self):
        """Tests the from_pytorch code path for training CrypTen models"""
        import torch.nn as nn
        import torch.nn.functional as F

        class ExampleNet(nn.Module):
            def __init__(self):
                super(ExampleNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
                self.fc1 = nn.Linear(16 * 13 * 13, 100)
                self.fc2 = nn.Linear(100, 2)

            def forward(self, x):
                out = self.conv1(x)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = F.relu(out)
                out = self.fc2(out)
                return out

        model_plaintext = ExampleNet()
        batch_size = 5
        x_orig = get_random_test_tensor(size=(batch_size, 1, 28, 28), is_float=True)
        y_orig = (
            get_random_test_tensor(size=(batch_size, 1), is_float=True).gt(0).long()
        )
        y_one_hot = onehot(y_orig, num_targets=2)

        # encrypt training sample:
        x_train = AutogradCrypTensor(crypten.cryptensor(x_orig))
        y_train = crypten.cryptensor(y_one_hot)
        dummy_input = torch.empty((1, 1, 28, 28))

        for loss_name in ["BCELoss", "CrossEntropyLoss", "MSELoss"]:
            # create loss function
            loss = getattr(crypten.nn, loss_name)()

            # create encrypted model
            model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
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

                # FIX check that any parameter with a non-zero gradient has changed??
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


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target test_mpc_benchmark)
if __name__ == "__main__":
    TestNN.benchmarks_enabled = True
    unittest.main()
