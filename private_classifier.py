#!/usr/bin/env python3


#
# This demonstrates how to train a classifier in the cloud on encrypted data.
# The cloud itself only sees encrypted data, but it does see real targets. An
# example use case could be: encrypted chat messages marked as spam or encrypted
# images as child pornography. Multiple clients can contribute data for training.
#
# The demo assumes that the clients are non-adversarial and collaborative.
#

import argparse
import logging
import os
import time

import crypten
import torch
import torch.distributed as dist
from crypten.common.encrypted_tensor import EncryptedTensor
from torchvision.datasets.mnist import MNIST


class PublicDownloader(object):
    """A context manager for downloading data from the public internet."""

    def __enter__(self):
        os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"
        os.environ["HTTPS_PROXY"] = "https://fwdproxy:8080"
        os.environ["http_proxy"] = "fwdproxy:8080"
        os.environ["https_proxy"] = "fwdproxy:8080"
        return self

    def __exit__(self, type, value, traceback):
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]
        del os.environ["http_proxy"]
        del os.environ["https_proxy"]


class CloudClassifier(object):
    """A binary classifier living in the cloud. The classifier can make
    predictions and be trained on both unencrypted and encrypted samples.

    This function can take batches of unencrypted data but only individual
    encrypted samples.
    """

    def __init__(self, num_dimensions, learning_rate=0.001, lr_decay=0.9999):
        super(CloudClassifier, self).__init__()
        self.weights = torch.randn(num_dimensions, 1).float().mul_(0.01)
        self.bias = torch.FloatTensor((1,)).fill_(0.0)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def predict(self, data):
        """Perform prediction of an encrypted or unencrypted data sample."""
        weights, bias = self.weights, self.bias
        if data.dim() == 1:
            data = data.view((1, data.nelement()))

        # compute and return scores:
        scores = data.matmul(weights).add_(bias)
        if hasattr(scores, "squeeze"):
            scores = scores.squeeze()
        return scores

    def update_parameters(self, data, target):
        """Update parameters based on a labeled, unencrypted data sample."""
        yx = data.mul(target)
        self.weights.add_(yx.view(yx.nelement(), 1).mul_(self.learning_rate))
        self.bias.add_(self.learning_rate * target)
        self.learning_rate *= self.lr_decay


def train(data, targets, num_dimensions, max_epochs=50):
    """Client(s) training linear classifier in the cloud on unencrypted or
    encrypted data (or a mix of both)."""

    # assertions:
    targets = targets.squeeze()
    assert targets.dim() == 1, "targets should be one-dimensional"
    assert len(data) == targets.nelement()

    # initialize classifier:
    classifier = CloudClassifier(num_dimensions)

    # perform training iterations:
    for epoch in range(max_epochs):
        loss_sum = 0.0
        for idx in torch.randperm(len(data)).tolist():

            # compute hinge loss:
            score = classifier.predict(data[idx])
            if isinstance(score, EncryptedTensor):
                score = score.get_plain_text()
            loss = 1.0 - score.item() * targets[idx].item()
            loss_sum += max(0.0, loss)

            # perform parameter update:
            if loss > 0.0:
                classifier.update_parameters(data[idx], targets[idx].item())

        # monitor progress:
        if (epoch + 1) % 10 == 0:
            logging.info("Epoch %d: loss = %2.5f" % (epoch + 1, loss_sum / len(data)))

    # return classifier:
    return classifier


def measure_error(predictions, targets):
    """Measure error of a set of predictions."""
    correct = predictions.mul(targets.float()).gt(0.0).long()
    return 1.0 - (correct.sum().item() / correct.nelement())


def load_dataset(max_size=None):
    """Loads binary classification problem on binary MNIST images."""

    # load training and test split:
    data, targets = {}, {}
    for split in ["train", "test"]:

        # download the MNIST dataset:
        with PublicDownloader():
            mnist = MNIST("/tmp", download=True, train=(split == "train"))

        # preprocess the MNIST dataset:
        subsample_size = 100
        idx = mnist.targets.eq(3) + mnist.targets.eq(8)
        split_data = mnist.data[idx, :][:subsample_size, :]
        split_labels = mnist.targets[idx][:subsample_size]
        data[split] = split_data.float().div_(255.0)
        data[split] = data[split].view(data[split].size(0), -1)
        targets[split] = split_labels.eq(8).long().mul(2).add_(-1)

        # reduce dataset size:
        if max_size is not None:
            data[split] = data[split][:max_size]
            targets[split] = targets[split][:max_size]

    # return:
    num_dimensions = data["train"].size(1)
    return data, targets, num_dimensions


def encrypt_data(data, protocol):
    """Encrypt data following the specified protocol."""
    if protocol == "spdz":
        return [crypten.MPCTensor(d) for d in data]
    elif protocol == "eivhe":
        return [crypten.EIVHETensor(d) for d in data]
    else:
        raise ValueError("Unsupported protocol %s." % protocol)


def main(args):
    """Demo of training a secure classifier."""

    # load MNIST dataset:
    logging.info("Loading MNIST dataset...")
    data, targets, num_dimensions = load_dataset(args.max_size)

    # train classifier:
    logging.info("Training classifier on unencrypted data...")
    classifier = train(data["train"], targets["train"], num_dimensions)
    predictions = classifier.predict(data["train"])
    train_error = measure_error(predictions, targets["train"])
    logging.info("Training error: %2.5f" % train_error)

    # measure test error on unencrypted data:
    logging.info(
        "Measuring unencrypted test error (on %d samples)..."
        % targets["test"].nelement()
    )
    predictions = classifier.predict(data["test"])

    # Logging full prediction value to track precision error
    logging.info("Unencrypted predictions: %s" % predictions)
    test_error = measure_error(predictions, targets["test"])
    logging.info("Unencrypted test error: %2.5f" % test_error)

    # performing encryption of test data:
    start = time.time()
    logging.info("Encrypting test data...")
    encrypted_data = encrypt_data(data["test"], args.protocol)

    # sync classifier
    dist.broadcast(classifier.weights, 0)
    dist.broadcast(classifier.bias, 0)

    # measure test error on encrypted data:
    logging.info(
        "Measuring encrypted test error (on %d samples)..." % targets["test"].nelement()
    )
    predictions = []
    for sample in encrypted_data:
        predictions.append(classifier.predict(sample).get_plain_text())
    predictions = torch.stack(predictions, dim=0)
    predictions = predictions.squeeze()
    logging.info("Encrypted predictions: %s" % predictions)
    end = time.time()

    test_error = measure_error(predictions, targets["test"])
    logging.info("Encrypted test error: %2.5f" % test_error)
    logging.info("Encryption and eval time (s) : %s" % (end - start))


# run all the things:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a classifier on encrypted data."
    )
    parser.add_argument(
        "--protocol",
        default="spdz",
        help="Privacy protocol can any of {'spdz', 'eivhe'}",
    )
    parser.add_argument(
        "--max_size", default=None, type=int, help="Maximum number of examples to use."
    )
    args = parser.parse_args()

    # set up logger:
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.basicConfig(format="| %(asctime)-15s | %(message)s")

    # let's go!
    main(args)
