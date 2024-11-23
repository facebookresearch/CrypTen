#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import numpy.linalg as nla
import torch


class NoopContextManager:
    """Context manager that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


def onehot(indices):
    """
    Converts index vector into one-hot matrix.
    """
    assert indices.dtype == torch.long, "indices must be long integers"
    assert indices.min() >= 0, "indices must be non-negative"
    onehot_vector = torch.zeros(
        indices.nelement(), indices.max() + 1, dtype=torch.uint8
    )
    onehot_vector.scatter_(1, indices.view(indices.nelement(), 1), 1)
    return onehot_vector


def kmeans_inference(data, clusters, hard=True, bandwidth=1.0):
    """
    Computes cluster assignments for a k-means clustering.
    """
    assert clusters.size(1) == data.size(
        1
    ), "cluster dimensionality does not match data dimensionality"

    # compute all pairwise distances:
    d2_sum = data.pow(2.0).sum(1, keepdim=True)
    c2_sum = clusters.pow(2.0).sum(1, keepdim=True)
    distances = data.matmul(clusters.t()).mul(-2.0).add_(d2_sum).add_(c2_sum.t())

    # compute assignments and return:
    if hard:
        assignments = distances.argmin(1)
        return assignments
    else:
        similarities = distances.mul_(-1.0 / (2.0 * bandwidth)).exp_()
        return similarities


def kmeans(data, K, max_iter=100):
    """
    Performs k-means clustering of data into K clusters.
    """
    assert K < data.size(0), "more clusters than data points"

    # initialize clusters at randomly selected data points:
    perm = torch.randperm(data.size(0))
    clusters = data[perm[:K], :]
    assignments = None
    for iter in range(max_iter):
        # compute assignments, and stop if converged:
        prev_assignments = assignments
        assignments = kmeans_inference(data, clusters)
        if prev_assignments is not None:
            num_changes = assignments.ne(prev_assignments).sum()
            logging.info(
                "K-means iteration %d: %d assignments changed" % (iter, num_changes)
            )
            if num_changes == 0:
                break

        # re-compute cluster means:
        for k in range(K):
            index = assignments == k
            if index.any():  # ignore empty clusters
                clusters[k, :] = data[index, :].mean(0)

    # done:
    return clusters


def pca(data, components):
    """
    Finds the `components` top principal components of the data.
    """
    assert components > 0 and components < data.size(1), "incorrect # of PCA dimensions"
    # We switch to numpy here as torch.symeig gave strange results.
    dtype = data.dtype
    data = data.numpy()
    data -= np.mean(data, axis=0, keepdims=True)
    cov = np.cov(data.T)
    L, V = nla.eigh(cov)
    return torch.tensor(V[:, -components:], dtype=dtype)


def process_mnist_files(raw_dir, processed_dir):
    """
    Uncompress zipped train and/or test image and label files, load the
    uncompressed data files, and save to .pt files so that datasets.MNIST
    can read it directly.
    """
    from torchvision import datasets

    os.makedirs(processed_dir, exist_ok=True)

    def extract_mnist_archive(data_file_name):
        """
        Extract the zipped data file and return the path to the uncompresse data
        file.
        If the zipped data file does not exist in raw_dir, it returns None.
        """
        data_file_archive = os.path.join(raw_dir, data_file_name + ".gz")
        if os.path.exists(data_file_archive):
            datasets.utils.extract_archive(data_file_archive, processed_dir)
            return os.path.join(processed_dir, data_file_name)
        else:
            return None

    train_image_file = extract_mnist_archive("train-images-idx3-ubyte")
    train_label_file = extract_mnist_archive("train-labels-idx1-ubyte")

    with open(os.path.join(processed_dir, datasets.MNIST.training_file), "wb") as f:
        if train_image_file and train_label_file:
            training_set = (
                datasets.mnist.read_image_file(train_image_file),
                datasets.mnist.read_label_file(train_label_file),
            )
            torch.save(training_set, f)

    test_image_file = extract_mnist_archive("t10k-images-idx3-ubyte")
    test_label_file = extract_mnist_archive("t10k-labels-idx1-ubyte")

    with open(os.path.join(processed_dir, datasets.MNIST.test_file), "wb") as f:
        if test_image_file and test_label_file:
            test_set = (
                datasets.mnist.read_image_file(test_image_file),
                datasets.mnist.read_label_file(test_label_file),
            )
            torch.save(test_set, f)
