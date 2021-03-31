#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile

import crypten
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from examples.meters import AccuracyMeter
from examples.util import NoopContextManager


try:
    from crypten.nn.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed
    SummaryWriter = None


def run_experiment(
    model_name,
    imagenet_folder=None,
    tensorboard_folder="/tmp",
    num_samples=None,
    context_manager=None,
):
    """Runs inference using specified vision model on specified dataset."""

    crypten.init()
    # check inputs:
    assert hasattr(models, model_name), (
        "torchvision does not provide %s model" % model_name
    )
    if imagenet_folder is None:
        imagenet_folder = tempfile.gettempdir()
        download = True
    else:
        download = False
    if context_manager is None:
        context_manager = NoopContextManager()

    # load dataset and model:
    with context_manager:
        model = getattr(models, model_name)(pretrained=True)
        model.eval()
        dataset = datasets.ImageNet(imagenet_folder, split="val", download=download)

    # define appropriate transforms:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    to_tensor_transform = transforms.ToTensor()

    # encrypt model:
    dummy_input = to_tensor_transform(dataset[0][0])
    dummy_input.unsqueeze_(0)
    encrypted_model = crypten.nn.from_pytorch(model, dummy_input=dummy_input)
    encrypted_model.encrypt()

    # show encrypted model in tensorboard:
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=tensorboard_folder)
        writer.add_graph(encrypted_model)
        writer.close()

    # loop over dataset:
    meter = AccuracyMeter()
    for idx, sample in enumerate(dataset):

        # preprocess sample:
        image, target = sample
        image = transform(image)
        image.unsqueeze_(0)
        target = torch.tensor([target], dtype=torch.long)

        # perform inference using encrypted model on encrypted sample:
        encrypted_image = crypten.cryptensor(image)
        encrypted_output = encrypted_model(encrypted_image)

        # measure accuracy of prediction
        output = encrypted_output.get_plain_text()
        meter.add(output, target)

        # progress:
        logging.info(
            "[sample %d of %d] Accuracy: %f" % (idx + 1, len(dataset), meter.value()[1])
        )
        if num_samples is not None and idx == num_samples - 1:
            break

    # print final accuracy:
    logging.info("Accuracy on all %d samples: %f" % (len(dataset), meter.value()[1]))
