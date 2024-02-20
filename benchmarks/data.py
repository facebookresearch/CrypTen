#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains data used for training / testing model benchmarks
"""

import os
from pathlib import Path

import PIL
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torchvision import transforms


class GaussianClusters:
    """Generates Glussian clusters for binary classes"""

    def __init__(self, n_samples=5000, n_features=20):
        self.n_samples = n_samples
        self.n_features = n_features
        x, x_test, y, y_test = GaussianClusters.generate_data(n_samples, n_features)
        self.x, self.y = x, y
        self.x_test, self.y_test = x_test, y_test

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


class Images:
    def __init__(self):
        self.x = self.preprocess_image()
        # image net 1k classes
        class_id = 463
        self.y = torch.tensor([class_id]).long()
        self.y_onehot = F.one_hot(self.y, 1000)
        self.x_test, self.y_test = self.x, self.y

    def preprocess_image(self):
        """Preprocesses sample image"""
        path = os.path.dirname(os.path.realpath(__file__))
        filename = "dog.jpg"
        input_image = PIL.Image.open(Path(os.path.join(path, filename)))
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
