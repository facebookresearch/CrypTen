# Contributing to CrypTen
We want to make contributing to this project as easy and transparent as
possible.

## Development Installation

1. Activate virtualenv with Python >= 3.7
2. Install Numpy and PyTorch Nightly
```bash
pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```
3. From your fork of the repo: `pip install -e .`

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License
By contributing to CrypTen, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
