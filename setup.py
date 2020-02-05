#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import sys

import setuptools


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crypten"))

# Read description and requirements.
with open("README.md", encoding="utf8") as f:
    readme = f.read()
with open("requirements.txt") as f:
    reqs = f.read()

# Set key package information.
DISTNAME = "crypten"
DESCRIPTION = "CrypTen: secure machine learning in PyTorch."
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = "MIT licensed, as found in the LICENSE file"
REQUIREMENTS = (reqs.strip().split("\n"),)

# Run installer.
if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        dependency_links=[],
        version="0.1",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/CrypTen",
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["pytest"],
    )
