Gradients
=========

``AutogradCrypTensors`` allow CrypTensors to store gradients and thus enable backpropagation.
Note CrypTen does not use the PyTorch optimizers.
Instead it directly implements stochastic gradient descent on encrypted data.
Using SGD in CrypTen is very similar to using PyTorch optimizers.

.. automodule:: crypten.autograd_cryptensor
    :members:
