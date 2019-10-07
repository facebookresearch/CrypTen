nn
==

``Crypten.nn`` provides modules and classes for defining and training neural networks similar to ``torch.nn``.

In CrypTen, encrypting a PyTorch network is straightforward:
first, we call the function ``from_pytorch`` that sets up a CrypTen network from the PyTorch network.
Then, we call ``encrypt`` on the CrypTen network to encrypt its parameters.
After encryption, the CrypTen network can also be decrypted.

.. note::

    In addition to the PyTorch network, the `from_pytorch` function also requires a dummy input of
    the shape of the model's input.
    The dummy input simply needs to be a `torch` tensor of the same shape; the values inside the tensor do not matter.
    (This is a requirement of `torch.distributed`, our communication backend.)

``Crypten.nn`` also provides several loss functions and
modules for building neural network layers.

.. automodule:: crypten.nn.loss
    :members:

.. automodule:: crypten.nn
    :members:
