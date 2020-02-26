#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate function and model benchmarks

To Run:
$ python benchmark.py

# save benchmarks to csv
$ python benchmark.py -p ~/Downloads/
"""

import argparse
import collections
import functools
import os
import timeit
from collections import namedtuple

import crypten
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


try:
    from . import models
except ImportError:
    # direct import if relative fails
    import models


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
        avg_runtime = np.sum(times) / n_loops
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        inner_quartile_range = q3_runtime - q1_runtime
        return avg_runtime, inner_quartile_range, return_val

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

    DOMAIN = torch.arange(start=0.01, end=100, step=0.01)
    # for exponential, sin, and cos
    TRUNCATED_DOMAIN = torch.arange(start=0.001, end=10, step=0.001)

    def __init__(self, tensor_size=(100, 100)):
        self.tensor_size = tensor_size
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
        return getattr(x, func)(y)

    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""
        x, y = torch.rand(self.tensor_size), torch.rand(self.tensor_size)
        x_enc, y_enc = crypten.cryptensor(x), crypten.cryptensor(y)

        Runtime = namedtuple("Runtime", "avg iqr")
        runtimes_plain_text, runtimes_crypten = [], []

        for func in FuncBenchmarks.UNARY + FuncBenchmarks.BINARY:
            second_operand, second_operand_enc = None, None
            if func in FuncBenchmarks.BINARY:
                second_operand, second_operand_enc = y, y_enc

            avg_runtime, iqr_runtime, _ = FuncBenchmarks.time_func(
                x, func, y=second_operand
            )
            runtime = Runtime(avg_runtime, iqr_runtime)
            runtimes_plain_text.append(runtime)

            avg_runtime, iqr_runtime, _ = FuncBenchmarks.time_func(
                x_enc, func, y=second_operand_enc
            )
            runtime = Runtime(avg_runtime, iqr_runtime)
            runtimes_crypten.append(runtime)

        return runtimes_plain_text, runtimes_crypten

    @staticmethod
    def calc_abs_error(ref, out):
        """Computes absolute error of encrypted output"""
        if ref.dtype == torch.bool:
            return (out == ref).sum().numpy()
        return torch.abs(out - ref).numpy()

    @staticmethod
    def calc_relative_error(ref, out):
        """Computes relative error of encrypted output"""
        if ref.dtype == torch.bool:
            return ((out == ref).sum() / ref.nelement()).numpy()
        errors = torch.abs((out - ref) / ref)
        # remove inf due to division by tiny numbers
        return errors[errors != float("inf")].numpy()

    def get_errors(self):
        """Computes the total error of approximations"""
        DOMAIN, TRUNCATED_DOMAIN = (
            FuncBenchmarks.DOMAIN,
            FuncBenchmarks.TRUNCATED_DOMAIN,
        )
        y = torch.rand(DOMAIN.shape)
        DOMAIN_enc, y_enc = crypten.cryptensor(DOMAIN), crypten.cryptensor(y)
        TRUNCATED_DOMAIN_enc = crypten.cryptensor(TRUNCATED_DOMAIN)
        abs_errors, relative_errors = [], []
        abs_errors_iqr, relative_errors_iqr = [], []

        for func in FuncBenchmarks.UNARY + FuncBenchmarks.BINARY:
            if func in ["exp", "cos", "sin"]:
                ref, out_enc = (
                    getattr(TRUNCATED_DOMAIN, func)(),
                    getattr(TRUNCATED_DOMAIN_enc, func)(),
                )
            elif func in FuncBenchmarks.UNARY:
                ref, out_enc = getattr(DOMAIN, func)(), getattr(DOMAIN_enc, func)()
            else:
                ref, out_enc = (
                    getattr(DOMAIN, func)(y),
                    getattr(DOMAIN_enc, func)(y_enc),
                )
            out = out_enc.get_plain_text()

            abs_error = FuncBenchmarks.calc_abs_error(ref, out)
            abs_errors.append(abs_error.sum())
            abs_errors_iqr.append(
                np.quantile(abs_error, 0.75) - np.quantile(abs_error, 0.25)
            )

            relative_error = FuncBenchmarks.calc_relative_error(ref, out)
            relative_errors.append(relative_error.sum())
            iqr = np.quantile(relative_error, 0.75) - np.quantile(relative_error, 0.25)
            relative_errors_iqr.append(iqr)

        return abs_errors, abs_errors_iqr, relative_errors, relative_errors_iqr

    def save(self, path):
        self.df.to_csv(os.path.join(path, "func_benchmarks.csv"), index=False)

    def run(self):
        """Runs and stores benchmarks in self.df"""
        crypten.init()
        runtimes_plain_text, runtimes_crypten = self.get_runtimes()

        abs_errors, abs_errors_iqr, relative_errors, relative_errors_iqr = (
            self.get_errors()
        )
        self.df = pd.DataFrame.from_dict(
            {
                "function": FuncBenchmarks.UNARY + FuncBenchmarks.BINARY,
                "runtime plain text": [r.avg for r in runtimes_plain_text],
                "runtime plain text IQR": [r.iqr for r in runtimes_plain_text],
                "runtime crypten": [r.avg for r in runtimes_crypten],
                "runtime crypten IQR": [r.iqr for r in runtimes_crypten],
                "abs error": abs_errors,
                "abs error IQR": abs_errors_iqr,
                "relative error": relative_errors,
                "relative error IQR": relative_errors_iqr,
            }
        )


class ModelBenchmarks:
    """Benchmarks runtime and accuracy of crypten models

    Models are benchmarked on synthetically generated
    Gaussian clusters for binary classification.

    Args:
        n_samples (int): number of samples for model training
        epochs (int): number of training epochs
        lr_rate (float): learning rate.
    """

    Model = collections.namedtuple("model", "name plain crypten")

    MODELS = [
        Model(
            name="logistic regression",
            plain=models.LogisticRegression,
            crypten=models.LogisticRegressionCrypTen,
        ),
        Model(
            name="feedforward neural network",
            plain=models.FeedForward,
            crypten=models.FeedForwardCrypTen,
        ),
    ]

    def __init__(self, n_samples=5000, n_features=20, epochs=50, lr_rate=0.1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.epochs = epochs
        self.lr_rate = lr_rate

        self.df = None

        data = ModelBenchmarks.generate_data(n_samples, n_features)
        self.x, self.x_test, self.y, self.y_test = data

        crypten.init()
        self.x_enc = crypten.cryptensor(self.x)
        self.y_enc = crypten.cryptensor(self.y)
        self.x_test_enc = crypten.cryptensor(self.x_test)

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    def generate_data(n_samples, n_features):
        """Generates Glussian clusters for binary classes

        Args:
            n_samples (int): number of samples
            n_features (int): number of features

        Returns: torch tensors with inputs and labels
        """
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            # by default, 2 features are redundant
            n_informative=n_features - 2,
            n_classes=2,
        )
        x = torch.tensor(x).float()
        y = torch.tensor(y).float().unsqueeze(-1)

        return train_test_split(x, y)

    @time_me(n_loops=1)
    def train(self, model):
        """Trains PyTorch binary classifier"""
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_rate)

        for _ in range(self.epochs):
            model.zero_grad()
            output = model(self.x)
            loss = criterion(output, self.y)
            loss.backward()
            optimizer.step()

        return model

    @time_me(n_loops=1)
    def train_crypten(self, model):
        """Trains crypten binary classifier"""
        criterion = crypten.nn.BCELoss()

        for _ in range(self.epochs):
            model.zero_grad()
            output = model(self.x_enc)
            loss = criterion(output, self.y_enc)
            loss.backward()
            model.update_parameters(self.lr_rate)

        return model

    def time_training(self):
        """Returns training time per epoch for plain text and CrypTen"""
        times_per_epoch = []
        times_per_epoch_crypten = []

        for model in ModelBenchmarks.MODELS:
            model_plain = model.plain(self.n_features)
            time_per_epoch = self.train(model_plain)[0] / self.epochs
            times_per_epoch.append(time_per_epoch)

            model_crypten = model.crypten(self.n_features)
            time_per_epoch_crypten = self.train_crypten(model_crypten)[0] / self.epochs
            times_per_epoch_crypten.append(time_per_epoch_crypten)

        return times_per_epoch, times_per_epoch_crypten

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
        accuracy = float((correct / y.shape[0]).numpy())
        return accuracy

    def evaluate(self):
        """Evaluates accuracy of crypten versus plain text models"""
        accuracies, accuracies_crypten = [], []

        for model in ModelBenchmarks.MODELS:
            model_plain = model.plain(self.n_features)
            _, _, model_plain = self.train(model_plain)
            accuracy = ModelBenchmarks.calc_accuracy(
                model_plain(self.x_test), self.y_test
            )
            accuracies.append(accuracy)

            model_crypten = model.crypten(self.n_features)
            _, _, model_crypten = self.train_crypten(model_crypten)
            output = model_crypten(self.x_test_enc).get_plain_text()
            accuracy = ModelBenchmarks.calc_accuracy(output, self.y_test)
            accuracies_crypten.append(accuracy)

        return accuracies, accuracies_crypten

    def save(self, path):
        self.df.to_csv(os.path.join(path, "model_benchmarks.csv"), index=False)

    def run(self):
        """Runs and stores benchmarks in self.df"""
        times_per_epoch, times_per_epoch_crypten = self.time_training()
        accuracies, accuracies_crypten = self.evaluate()
        model_names = [model.name for model in ModelBenchmarks.MODELS]
        n_rows = len(times_per_epoch)
        self.df = pd.DataFrame.from_dict(
            {
                "model": model_names + model_names,
                "seconds per epoch": times_per_epoch + times_per_epoch_crypten,
                "is plain text": [True] * n_rows + [False] * n_rows,
                "accuracy": accuracies + accuracies_crypten,
            }
        )
        self.df = self.df.sort_values(by="model")


def parse_path():
    """Parses path from command line argument"""
    parser = argparse.ArgumentParser(description="Benchmark Functions")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=False,
        default=None,
        help="path to save function benchmarks",
    )
    args = parser.parse_args()
    return args.path


if __name__ == "__main__":
    path = parse_path()

    func_benchmarks = FuncBenchmarks()
    func_benchmarks.run()
    print(func_benchmarks)

    model_benchmarks = ModelBenchmarks()
    model_benchmarks.run()
    print(model_benchmarks)

    if path:
        func_benchmarks.save(path)
        model_benchmarks.save(path)
