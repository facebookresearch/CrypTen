#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import io
import logging
import unittest

import crypten
import torch
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.nn import onnx_converter
from test.multiprocess_test_case import (
    get_random_test_tensor,
    MultiProcessTestCase,
    onehot,
)


class TestOnnxConverter:
    """Tests PyTorch and Tensorflow model imports"""

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

    def _check_reference_parameters(self, init_name, reference, model):
        for name, param in model.named_parameters(recurse=False):
            local_name = init_name + "_" + name
            self._check(param, reference[local_name], "parameter update failed")
        for name, module in model._modules.items():
            local_name = init_name + "_" + name
            self._check_reference_parameters(local_name, reference, module)

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

    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communicator
        if self.rank >= 0:
            crypten.init()

    """
    @unittest.skip("CrypTen no longer supports from_tensorflow")
    def test_tensorflow_model_conversion(self) -> None:
        import tensorflow as tf
        import tf2onnx

        # create simple model
        model_tf1 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    10,
                    activation=tf.nn.relu,
                    kernel_initializer="ones",
                    bias_initializer="ones",
                    input_shape=(4,),
                ),
                tf.keras.layers.Dense(
                    10,
                    activation=tf.nn.relu,
                    kernel_initializer="ones",
                    bias_initializer="ones",
                ),
                tf.keras.layers.Dense(3, kernel_initializer="ones"),
            ]
        )

        model_tf2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    3,
                    activation="relu",
                    strides=1,
                    kernel_initializer="ones",
                    bias_initializer="ones",
                    input_shape=(32, 32, 3),
                ),
                tf.keras.layers.MaxPooling2D(3),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.5),
            ]
        )

        model_tf3 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    32,
                    1,
                    activation="relu",
                    strides=1,
                    kernel_initializer="ones",
                    bias_initializer="ones",
                    input_shape=(6, 128),
                ),
                tf.keras.layers.AvgPool1D(1),
            ]
        )

        feature_sizes = [(1, 4), (1, 32, 32, 3), (1, 6, 128)]
        label_sizes = [(1, 3), (1, 32), (1, 6, 32)]

        for i, curr_model_tf in enumerate([model_tf1, model_tf2, model_tf3]):
            # create a random feature vector
            features = get_random_test_tensor(
                size=feature_sizes[i], is_float=True, min_value=1, max_value=3
            )
            labels = get_random_test_tensor(
                size=label_sizes[i], is_float=True, min_value=1
            )

            # convert to a TF tensor via numpy
            features_tf = tf.convert_to_tensor(features.numpy())
            labels_tf = tf.convert_to_tensor(labels.numpy())
            # compute the tensorflow predictions
            curr_model_tf.compile("sgd", loss=tf.keras.losses.MeanSquaredError())
            curr_model_tf.fit(features_tf, labels_tf)
            result_tf = curr_model_tf(features_tf, training=False)

            # convert TF model to CrypTen model
            # write as a SavedModel, then load GraphDef from it
            import tempfile

            saved_model_dir = tempfile.NamedTemporaryFile(delete=True).name
            os.makedirs(saved_model_dir, exist_ok=True)
            curr_model_tf.save(saved_model_dir)
            graph_def, inputs, outputs = tf2onnx.tf_loader.from_saved_model(
                saved_model_dir, None, None
            )

            model_enc = crypten.nn.from_tensorflow(graph_def, inputs, outputs)

            # encrypt model and run it
            model_enc.encrypt()
            features_enc = crypten.cryptensor(features)
            result_enc = model_enc(features_enc)

            # compare the results
            result = torch.tensor(result_tf.numpy())
            self._check(result_enc, result, "nn.from_tensorflow failed")
    """

    def test_from_pytorch_training_classification(self):
        """Tests from_pytorch CrypTen training for classification models"""
        import torch.nn as nn
        import torch.nn.functional as F

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
                self.fc1 = nn.Linear(16 * 13 * 13, 100)
                self.fc2 = nn.Linear(100, 2)

            def forward(self, x):
                out = self.conv1(x)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = out.view(-1, 16 * 13 * 13)
                out = self.fc1(out)
                out = F.relu(out)
                out = self.fc2(out)
                out = F.softmax(out, dim=1)
                return out

        model_plaintext = CNN()
        batch_size = 5
        x_orig = get_random_test_tensor(size=(batch_size, 1, 28, 28), is_float=True)
        y_orig = (
            get_random_test_tensor(size=(batch_size, 1), is_float=True).gt(0).long()
        )
        y_one_hot = onehot(y_orig, num_targets=2)

        # encrypt training sample:
        x_train = crypten.cryptensor(x_orig, requires_grad=True)
        y_train = crypten.cryptensor(y_one_hot)
        dummy_input = torch.empty((1, 1, 28, 28))

        for loss_name in ["BCELoss", "CrossEntropyLoss"]:
            # create encrypted model
            model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
            model.train()
            model.encrypt()

            self._check_training(model, x_train, y_train, loss_name)

        self._check_model_export(model, x_train)

    def test_from_pytorch_training_regression(self):
        """Tests from_pytorch CrypTen training for regression models"""
        import torch.nn as nn
        import torch.nn.functional as F

        class FeedForward(nn.Module):
            def __init__(self):
                super(FeedForward, self).__init__()
                self.fc1 = nn.Linear(3, 10)
                self.fc2 = nn.Linear(10, 1)

            def forward(self, x):
                out = self.fc1(x)
                out = F.relu(out)
                out = self.fc2(out)
                return out

        model_plaintext = FeedForward()
        batch_size = 5

        x_orig = get_random_test_tensor(size=(batch_size, 3), is_float=True)
        dummy_input = torch.empty((1, 3))
        # y is a linear combo of features 1 and 3
        y_orig = 2 * x_orig[:, 0] + 3 * x_orig[:, 2]

        x_train = crypten.cryptensor(x_orig, requires_grad=True)
        y_train = crypten.cryptensor(y_orig.unsqueeze(-1))

        # create encrypted model
        model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
        model.train()
        model.encrypt()

        self._check_training(model, x_train, y_train, "MSELoss")
        self._check_model_export(model, x_train)

    def _check_training(
        self, model, x_train, y_train, loss_name, num_epochs=2, learning_rate=0.001
    ):
        """Verifies gradient updates and loss decreases during training"""
        # create loss function
        loss = getattr(crypten.nn, loss_name)()

        for i in range(num_epochs):
            output = model(x_train)
            loss_value = loss(output, y_train)

            # set gradients to "zero"
            model.zero_grad()
            for param in model.parameters():
                self.assertIsNone(param.grad, "zero_grad did not reset gradients")

            # perform backward pass
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

            # check parameter update
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
            f"{loss_name} has not decreased after training",
        )

    def _check_model_export(self, crypten_model, x_enc):
        """Checks that exported model returns the same results as crypten model"""
        pytorch_model = crypten_model.decrypt().to_pytorch()
        x_plain = x_enc.get_plain_text()

        y_plain = pytorch_model(x_plain)
        crypten_model.encrypt()
        y_enc = crypten_model(x_enc)

        self._check(y_enc, y_plain, msg="Model export failed.")

    def test_get_operator_class(self):
        """Checks operator is a valid crypten module"""
        Node = collections.namedtuple("Node", "op_type")

        op_types = ["Sum", "AveragePool", "Mean"]
        for op_type in op_types:
            node = Node(op_type)
            operator = onnx_converter._get_operator_class(node.op_type, {})
            self.assertTrue(
                issubclass(operator, crypten.nn.Module),
                f"{op_type} operator class {operator} is not a CrypTen module.",
            )
        # check conv
        kernel_shapes = [[1], [3, 3]]
        node = Node("Conv")
        for kernel_shape in kernel_shapes:
            attributes = {"kernel_shape": kernel_shape}
            operator = onnx_converter._get_operator_class(node.op_type, attributes)

        # check invalid op_types
        invalid_types = [("Convolution", {"kernel_shape": [3, 3, 3]}), ("Banana", {})]
        for invalid_type, attr in invalid_types:
            with self.assertRaises(ValueError):
                node = Node(invalid_type)
                operator = onnx_converter._get_operator_class(node.op_type, attr)

    def test_export_pytorch_model(self):
        """Tests loading of onnx model from a file"""
        pytorch_model = PyTorchLinear()
        dummy_input = torch.empty(10, 10)

        with io.BytesIO() as f:
            onnx_converter._export_pytorch_model(f, pytorch_model, dummy_input)

    def test_from_onnx(self):
        """Tests construction of crypten model from onnx graph"""
        pytorch_model = PyTorchLinear()
        dummy_input = torch.empty(10, 10)

        with io.BytesIO() as f:
            f = onnx_converter._export_pytorch_model(f, pytorch_model, dummy_input)
            f.seek(0)

            crypten_model = onnx_converter.from_onnx(f)

        self.assertTrue(hasattr(crypten_model, "encrypt"))

    def test_reshape_plain_text_conversion(self):
        """Verifies shape inputs in reshape are properly imported"""

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # (1, 4) is stored in a constant module
                out = x.reshape(1, 4)
                out = self.fc1(out)
                return out

        model = Net()
        x = torch.ones(2, 2)
        x_enc = crypten.cryptensor(x)
        y = model(x)
        model_crypten = onnx_converter.from_pytorch(model, torch.empty(x.shape))

        model_crypten.encrypt()
        y_enc = model_crypten(x_enc)
        self.assertTrue(y_enc.shape == y.shape)


class PyTorchLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestOnnxConverter):
    def setUp(self) -> None:
        self._original_provider = cfg.mpc.provider
        cfg.mpc.provider = "TFP"
        super(TestTFP, self).setUp()

    def tearDown(self) -> None:
        cfg.mpc.provider = self._original_provider
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestOnnxConverter):
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
