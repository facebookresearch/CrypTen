#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate function and model benchmarks

To Run:
$ python benchmark.py

# only function benchmarks
$ python benchmark.py --only-functions
$ python benchmark.py --only-functions --world-size 2

# benchmark functions and all models
$ python benchmark.py --advanced-models

# save benchmarks to csv
$ python benchmark.py -p ~/Downloads/
"""

import argparse
import functools
import os
import timeit
from collections import namedtuple

import crypten
import crypten.communicator as comm
import numpy as np
import pandas as pd
import torch
from examples import multiprocess_launcher


try:
    from . import data
    from . import models
except ImportError:
    # direct import if relative fails
    import data
    import models

Runtime = namedtuple("Runtime", "mid q1 q3")


def time_me(func=None, n_loops=10):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        return_val = func(*args, **kwargs)
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime)
        return runtime, return_val

    return timing_wrapper


class FuncBenchmarks:
    """Benchmarks runtime and error of crypten functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    BINARY = ["add", "sub", "mul", "matmul", "gt", "lt", "eq"]

    UNARY = [
        "sigmoid",
        "relu",
        "tanh",
        "exp",
        "log",
        "reciprocal",
        "cos",
        "sin",
        "sum",
        "mean",
        "neg",
    ]

    LAYERS = ["conv2d"]

    DOMAIN = torch.arange(start=0.01, end=100, step=0.01)
    # for exponential, sin, and cos
    TRUNCATED_DOMAIN = torch.arange(start=0.001, end=10, step=0.001)

    def __init__(self, tensor_size=(100, 100), device=None):
        self.tensor_size = tensor_size
        self.device = device

        crypten.init(device=device)
        # dataframe for benchmarks
        self.df = None

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_func(x, func, y=None):
        """Invokes func as a method of x"""
        if y is None:
            return getattr(x, func)()

        if func in {"conv1d", "conv2d"}:
            if torch.is_tensor(x):
                return getattr(torch.nn.functional, func)(x, y)
            return getattr(x, func)(y)

        return getattr(x, func)(y)

    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""
        x, y = (
            torch.rand(self.tensor_size, device=self.device),
            torch.rand(self.tensor_size, device=self.device),
        )
        x_enc, y_enc = crypten.cryptensor(x), crypten.cryptensor(y)

        runtimes, runtimes_enc = [], []

        for func in FuncBenchmarks.UNARY + FuncBenchmarks.BINARY:
            second_operand, second_operand_enc = None, None
            if func in FuncBenchmarks.BINARY:
                second_operand, second_operand_enc = y, y_enc

            runtime, _ = FuncBenchmarks.time_func(x, func, y=second_operand)
            runtimes.append(runtime)

            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, func, y=second_operand_enc)
            runtimes_enc.append(runtime_enc)

        # add layer runtimes
        runtime_layers, runtime_layers_enc = self.get_layer_runtimes()
        runtimes.extend(runtime_layers)
        runtimes_enc.extend(runtime_layers_enc)

        return runtimes, runtimes_enc

    def get_layer_runtimes(self):
        """Returns runtimes for layers"""

        runtime_layers, runtime_layers_enc = [], []

        for layer in FuncBenchmarks.LAYERS:
            if layer == "conv1d":
                x, x_enc, y, y_enc = self.random_conv1d_inputs()
            elif layer == "conv2d":
                x, x_enc, y, y_enc = self.random_conv2d_inputs()
            else:
                raise ValueError(f"{layer} not supported")

            runtime, _ = FuncBenchmarks.time_func(x, layer, y=y)
            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, layer, y=y_enc)

            runtime_layers.append(runtime)
            runtime_layers_enc.append(runtime_enc)

        return runtime_layers, runtime_layers_enc

    def random_conv2d_inputs(self):
        """Returns random input and weight tensors for 2d convolutions"""
        filter_size = [size // 10 for size in self.tensor_size]
        x_conv2d = torch.rand(1, 1, *self.tensor_size, device=self.device)
        weight2d = torch.rand(1, 1, *filter_size, device=self.device)
        x_conv2d_enc = crypten.cryptensor(x_conv2d)
        weight2d_enc = crypten.cryptensor(weight2d)
        return x_conv2d, x_conv2d_enc, weight2d, weight2d_enc

    def random_conv1d_inputs(self):
        """Returns random input and weight tensors for 1d convolutions"""
        size = self.tensor_size[0]
        filter_size = size // 10
        (x_conv1d,) = torch.rand(1, 1, size, device=self.device)
        weight1d = torch.rand(1, 1, filter_size, device=self.device)
        x_conv1d_enc = crypten.cryptensor(x_conv1d)
        weight1d_enc = crypten.cryptensor(weight1d)
        return x_conv1d, x_conv1d_enc, weight1d, weight1d_enc

    @staticmethod
    def calc_abs_error(ref, out):
        """Computes total absolute error"""
        if ref.dtype == torch.bool:
            errors = (out != ref).sum().cpu().numpy()
            return errors
        errors = torch.abs(out - ref).cpu().numpy()
        return errors.sum()

    @staticmethod
    def calc_relative_error(ref, out):
        """Computes average relative error"""
        if ref.dtype == torch.bool:
            errors = ((out != ref).sum() // ref.nelement()).cpu().numpy()
            return errors
        errors = torch.abs((out - ref) / ref)
        # remove inf due to division by tiny numbers
        errors = errors[errors != float("inf")].cpu().numpy()
        return errors.mean()

    def call_function_on_domain(self, func):
        """Call plain text and CrypTen function on given function
        Uses DOMAIN, TRUNCATED_DOMAIN, or appropriate layer inputs

        Returns: tuple of (plain text result, encrypted result)
        """
        DOMAIN, TRUNCATED_DOMAIN = (
            FuncBenchmarks.DOMAIN,
            FuncBenchmarks.TRUNCATED_DOMAIN,
        )
        DOMAIN, TRUNCATED_DOMAIN = (
            DOMAIN.to(self.device),
            TRUNCATED_DOMAIN.to(self.device),
        )
        y = torch.rand(DOMAIN.shape, device=self.device)
        DOMAIN_enc, y_enc = crypten.cryptensor(DOMAIN), crypten.cryptensor(y)
        TRUNCATED_DOMAIN_enc = crypten.cryptensor(TRUNCATED_DOMAIN)

        if func in ["exp", "cos", "sin"]:
            ref, out_enc = (
                getattr(TRUNCATED_DOMAIN, func)(),
                getattr(TRUNCATED_DOMAIN_enc, func)(),
            )
        elif func in FuncBenchmarks.UNARY:
            ref, out_enc = getattr(DOMAIN, func)(), getattr(DOMAIN_enc, func)()
        elif func in FuncBenchmarks.LAYERS:
            ref, out_enc = self._call_layer(func)
        elif func in FuncBenchmarks.BINARY:
            ref, out_enc = (getattr(DOMAIN, func)(y), getattr(DOMAIN_enc, func)(y_enc))
        else:
            raise ValueError(f"{func} not supported")

        return ref, out_enc

    def get_errors(self):
        """Computes the total error of approximations"""
        abs_errors, relative_errors = [], []

        functions = FuncBenchmarks.UNARY + FuncBenchmarks.BINARY
        functions += FuncBenchmarks.LAYERS

        for func in functions:
            ref, out_enc = self.call_function_on_domain(func)
            out = out_enc.get_plain_text()

            abs_error = FuncBenchmarks.calc_abs_error(ref, out)
            abs_errors.append(abs_error)

            relative_error = FuncBenchmarks.calc_relative_error(ref, out)
            relative_errors.append(relative_error)

        return abs_errors, relative_errors

    def _call_layer(self, layer):
        """Call supported layers"""
        if layer == "conv1d":
            x, x_enc, y, y_enc = self.random_conv1d_inputs()
        elif layer == "conv2d":
            x, x_enc, y, y_enc = self.random_conv2d_inputs()
        else:
            raise ValueError(f"{layer} not supported")

        ref = getattr(torch.nn.functional, layer)(x, y)
        out_enc = getattr(x_enc, layer)(y_enc)

        return ref, out_enc

    def save(self, path):
        self.df.to_csv(os.path.join(path, "func_benchmarks.csv"), index=False)

    def run(self):
        """Runs and stores benchmarks in self.df"""
        runtimes, runtimes_enc = self.get_runtimes()

        abs_errors, relative_errors = self.get_errors()

        self.df = pd.DataFrame.from_dict(
            {
                "function": FuncBenchmarks.UNARY
                + FuncBenchmarks.BINARY
                + FuncBenchmarks.LAYERS,
                "runtime": [r.mid for r in runtimes],
                "runtime Q1": [r.q1 for r in runtimes],
                "runtime Q3": [r.q3 for r in runtimes],
                "runtime crypten": [r.mid for r in runtimes_enc],
                "runtime crypten Q1": [r.q1 for r in runtimes_enc],
                "runtime crypten Q3": [r.q3 for r in runtimes_enc],
                "total abs error": abs_errors,
                "average relative error": relative_errors,
            }
        )


class ModelBenchmarks:
    """Benchmarks runtime and accuracy of crypten models

    Models are benchmarked on synthetically generated
    Gaussian clusters for binary classification. Resnet18 is
    benchmarks use image data.

    Args:
        n_samples (int): number of samples for Gaussian cluster model training
        n_features (int): number of features for the Gaussian clusters.
        epochs (int): number of training epochs
        lr_rate (float): learning rate.
    """

    def __init__(self, device=None, advanced_models=False):
        self.df = None
        self.device = device

        crypten.init(device=device)

        self.models = models.MODELS
        if not advanced_models:
            self.remove_advanced_models()

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Model Benchmarks"

    def remove_advanced_models(self):
        """Removes advanced models from instance"""
        self.models = list(filter(lambda x: not x.advanced, self.models))

    @time_me(n_loops=3)
    def train(self, model, x, y, epochs, lr, loss):
        """Trains PyTorch model

        Args:
            model (PyTorch model): model to be trained
            x (torch.tensor): inputs
            y (torch.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, torch.nn.Module), "must be a PyTorch model"
        criterion = getattr(torch.nn, loss)()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        x, y = x.to(self.device), y.to(self.device)
        model = model.to(self.device)

        for _ in range(epochs):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return model

    @time_me(n_loops=3)
    def train_crypten(self, model, x, y, epochs, lr, loss):
        """Trains crypten encrypted model

        Args:
            model (CrypTen model): model to be trained
            x (crypten.tensor): inputs
            y (crypten.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, crypten.nn.Module), "must be a CrypTen model"
        criterion = getattr(crypten.nn, loss)()

        x, y = x.to(self.device), y.to(self.device)

        model = model.to(self.device)
        model = model.encrypt()

        for _ in range(epochs):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            model.update_parameters(lr)

        return model

    def time_training(self):
        """Returns training time per epoch for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            x, y = model.data.x, model.data.y
            model_plain = model.plain()
            runtime, _ = self.train(model_plain, x, y, 1, model.lr, model.loss)
            runtimes.append(runtime)

            x_enc, y_enc = model.data.x_enc, model.data.y_enc
            x_enc, y_enc = x_enc.to(self.device), y_enc.to(self.device)
            model_enc = model.crypten()
            runtime_enc, _ = self.train_crypten(
                model_enc, x_enc, y_enc, 1, model.lr, model.loss
            )
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    @time_me(n_loops=3)
    def predict(self, model, x):
        x = x.to(self.device)
        model = model.to(self.device)

        if isinstance(model, crypten.nn.Module):
            model.encrypt()

        y = model(x)
        return y

    def time_inference(self):
        """Returns inference time for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            model_plain = model.plain()
            runtime, _ = self.predict(model_plain, model.data.x)
            runtimes.append(runtime)

            model_enc = model.crypten()
            runtime_enc, _ = self.predict(model_enc, model.data.x_enc)
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    @staticmethod
    def calc_accuracy(output, y, threshold=0.5):
        """Computes percent accuracy

        Args:
            output (torch.tensor): model output
            y (torch.tensor): true label
            threshold (float): classification threshold

        Returns (float): percent accuracy
        """
        predicted = (output > threshold).float()
        correct = (predicted == y).sum().float()
        accuracy = float((correct / y.shape[0]).cpu().numpy())
        return accuracy

    def evaluate(self):
        """Evaluates accuracy of crypten versus plain text models"""
        accuracies, accuracies_crypten = [], []

        for model in self.models:
            model_plain = model.plain()
            x, y = model.data.x, model.data.y
            _, model_plain = self.train(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )

            x_test, y_test = model.data.x_test, model.data.y_test
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            accuracy = ModelBenchmarks.calc_accuracy(model_plain(x_test), y_test)
            accuracies.append(accuracy)

            model_crypten = model.crypten()
            x_enc, y_enc = model.data.x_enc, model.data.y_enc
            _, model_crypten = self.train_crypten(
                model_crypten, x_enc, y_enc, model.epochs, model.lr, model.loss
            )
            x_test_enc = model.data.x_test_enc.to(self.device)

            output = model_crypten(x_test_enc).get_plain_text()
            accuracy = ModelBenchmarks.calc_accuracy(output, y_test)
            accuracies_crypten.append(accuracy)

        return accuracies, accuracies_crypten

    def save(self, path):
        self.df.to_csv(os.path.join(path, "model_benchmarks.csv"), index=False)

    def run(self):
        """Runs and stores benchmarks in self.df"""
        training_runtimes, training_runtimes_enc = self.time_training()
        inference_runtimes, inference_runtimes_enc = self.time_inference()
        accuracies, accuracies_crypten = self.evaluate()
        model_names = [model.name for model in self.models]

        training_times_both = training_runtimes + training_runtimes_enc
        inference_times_both = inference_runtimes + inference_runtimes_enc

        half_n_rows = len(training_runtimes)
        self.df = pd.DataFrame.from_dict(
            {
                "model": model_names + model_names,
                "seconds per epoch": [t.mid for t in training_times_both],
                "seconds per epoch q1": [t.q1 for t in training_times_both],
                "seconds per epoch q3": [t.q3 for t in training_times_both],
                "inference time": [t.mid for t in inference_times_both],
                "inference time q1": [t.q1 for t in inference_times_both],
                "inference time q3": [t.q3 for t in inference_times_both],
                "is plain text": [True] * half_n_rows + [False] * half_n_rows,
                "accuracy": accuracies + accuracies_crypten,
            }
        )
        self.df = self.df.sort_values(by="model")


def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Functions")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=False,
        default=None,
        help="path to save function benchmarks",
    )
    parser.add_argument(
        "--only-functions",
        "-f",
        required=False,
        default=False,
        action="store_true",
        help="run only function benchmarks",
    )
    parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        required=False,
        default=1,
        help="world size for number of parties",
    )
    parser.add_argument(
        "--advanced-models",
        required=False,
        default=False,
        action="store_true",
        help="run advanced model (resnet, transformer, etc.) benchmarks",
    )
    parser.add_argument(
        "--device",
        required=False,
        default=None,
        help="the device to run the benchmarks",
    )
    args = parser.parse_args()
    return args


def multiprocess_caller(args):
    """Runs multiparty benchmarks and prints/saves from source 0"""
    for benchmark in args.benchmarks:
        benchmark.run()
        rank = comm.get().get_rank()
        if rank == 0:
            pd.set_option("display.precision", 3)
            if args.path:
                benchmark.save(args.path)


def main():
    """Runs benchmarks and saves if path is provided"""
    args = get_args()
    benchmarks = [
        FuncBenchmarks(device=args.device),
        ModelBenchmarks(device=args.device, advanced_models=args.advanced_models),
    ]

    if args.only_functions:
        benchmarks = [FuncBenchmarks()]

    if args.world_size > 1:
        args.benchmarks = benchmarks
        launcher = multiprocess_launcher.MultiProcessLauncher(
            args.world_size, multiprocess_caller, fn_args=args
        )
        launcher.start()
        launcher.join()
        launcher.terminate()

    else:
        pd.set_option("display.precision", 3)
        for benchmark in benchmarks:
            benchmark.run()
            if args.path:
                benchmark.save(args.path)


if __name__ == "__main__":
    main()
