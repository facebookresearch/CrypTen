<p align="center"><img width="70%" src="docs/\_static/img/CrypTen_Identity_Horizontal_Lockup_01_FullColor.png" alt="CrypTen logo" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/CrypTen/blob/master/LICENSE) [![CircleCI](https://circleci.com/gh/facebookresearch/CrypTen.svg?style=shield)](https://circleci.com/gh/facebookresearch/CrypTen/tree/master) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/CrypTen/blob/master/CONTRIBUTING.md)

--------------------------------------------------------------------------------

CrypTen is a framework for Privacy Preserving Machine Learning built on PyTorch.
Its goal is to make secure computing techniques accessible to Machine Learning practitioners.
It currently implements [Secure Multiparty Computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
as its secure computing backend and offers three main benefits to ML researchers:

1. It is machine learning first. The framework presents the protocols via a `CrypTensor`
   object that looks and feels exactly like a PyTorch `Tensor`. This allows the user to use
   automatic differentiation and neural network modules akin to those in PyTorch.

2. CrypTen is library-based. It implements a tensor library just as PyTorch does.
   This makes it easier for practitioners to debug, experiment on, and explore ML models.

3. The framework is built with real-world challenges in mind. CrypTen does not scale back or
   oversimplify the implementation of the secure protocols.

Here is a bit of CrypTen code that encrypts and decrypts tensors and adds them

```python
import torch
import crypten

crypten.init()

x = torch.tensor([1.0, 2.0. 3.0])
x_enc = crypten.cryptensor(x) # encrypt

x_dec = x_enc.get_plain_text() # decrypt

y_enc = crypten.cryptensor([2.0, 3.0, 4.0])
sum_xy = x_enc + y_enc # add encrypted tensors
sum_xy_dec = sum_xy.get_plain_text() # decrypt sum
```

It is currently not production ready and its main use is as a research framework.

## Installing CrypTen

CrypTen currently runs on Linux and Mac. It also needs a PyTorch nightly build.
Windows is not supported. We also do not currently support computation on GPUs.

Install Anaconda 2019.07 or later and then do the following:

_For Linux or Mac_
```bash
conda create -n crypten-env python=3.7
conda activate crypten-env
conda install pytorch torchvision -c pytorch
git clone https://github.com/facebookresearch/CrypTen.git
cd CrypTen
pip install -e .
```

If you want to run the examples in the `examples` directory, you should also do the following
```bash
pip install -r requirements.examples.txt
```

## Examples

We provide examples covering a range of models in the `examples` directory

1. The linear SVM example, `mpc_linear_svm`, generates random data and trains a
  SVM classifier on encrypted data.
2. The LeNet example, `mpc_cifar`, trains an adaptation of LeNet on CIFAR in
  cleartext and encrypts the model and data for inference.
3. The TFE benchmark example, `tfe_benchmarks`, trains three different network
  architectures on MNIST in cleartext, and encrypts the trained model and data
  for inference.
4. The bandits example, `bandits`, trains a contextual bandits model on
  encrypted data (MNIST).
5. The imagenet example, `mpc_imagenet`, performs inference on pretrained
  models from `torchvision`.

For examples that train in cleartext, we also provide pre-trained models in
cleartext in the `model` subdirectory of each example subdirectory.

You can check all example specific command line options by doing the following;
shown here for `tfe_benchmarks`:

```bash
    $ python3 examples/tfe_benchmarks/launcher.py --help
```

## How CrypTen works

We have a set of tutorials in the `tutorials` directory to show how
CrypTen works. These are presented as Jupyter notebooks so please install
the following in your conda environment

```bash
conda install ipython jupyter
pip install -r requirements.examples.txt
```

1. `Introduction.ipynb` - an introduction to Secure Multiparty Compute; CrypTen's
   underlying secure computing protocol; use cases we are trying to solve and the
   threat model we assume.
2. `Tutorial_1_Basics_of_CrypTen_Tensors.ipynb` - introduces `CrypTensor`, CrypTen's
   encrypted tensor object, and shows how to use it to do various operations on
   this object.
3. `Tutorial_2_Inside_CrypTensors.ipynb` – delves deeper into `CrypTensor` to show
   the inner workings; specifically how `CrypTensor` uses `MPCTensor` for its
   backend and the two different kind of _sharings_, arithmetic and binary, are
   used for two different kind of functions. It also shows CrypTen's
   [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)-inspired
   programming model.
4. `Tutorial_3_Introduction_to_Access_Control.ipynb` - shows how to train a linear
   model using CrypTen and shows various scenarios of data labeling, feature
   aggregation, dataset augmentation and model hiding where this is applicable.
5. `Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb` – shows how
   CrypTen can load a pre-trained PyTorch model, encrypt it and then do inference
   on encrypted data.
6. `Tutorial_5_Under_the_hood_of_Encrypted_Networks.ipynb` - examines how CrypTen
   loads PyTorch models, how they are encrypted and how data moves through a multilayer
   network.
7. `Tutorial_6_CrypTen_on_AWS_instances.ipynb` - shows how to use `scrips/aws_launcher.py`
   to launch our examples on AWS. It can also work with your code written in CrypTen.
8. `Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb` - introduces `AutogradCrypTensor`,
   a wrapper that adds automatic differentiation functionality to `CrypTensor`. This
   allows you to train neural networks in CrypTen. We expect to move this functionality
   into the `CrypTensor` object in a future release.


## Documentation
CrypTen is documented [here](https://crypten.readthedocs.io/en/latest/)

## Join the CrypTen community
Please contact [us](mailto:ssengupta@fb.com) to join the CrypTen community on [Slack](https://cryptensor.slack.com)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
CrypTen is MIT licensed, as found in the LICENSE file.
