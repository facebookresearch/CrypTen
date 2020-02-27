CrypTensor
==========

A ``CrypTensor`` is an encrypted ``torch`` tensor for secure computations.

CrypTen currently only supports secure MPC protocols
(though we intend to add support for other advanced encryption protocols).
Using the :doc:`mpctensor` backend, a ``CrypTensor`` acts as ``torch`` tensor
whose values are encrypted using the secure MPC protocol.

To create a cryptensor,

.. code-block:: python

    # Create torch tensor
    x = torch.tensor([1.0, 2.0, 3.0])

    # Encrypt x
    x_enc = crypten.cryptensor(x)

We can decrypt ``x_enc`` by calling ``x_enc.get_plain_text()``.

Tensor Operations
-----------------

``CrypTensor``s provide various operations similar to ``torch`` tensors.

.. automodule:: crypten.cryptensor
    :members:

Autograd
--------
``CrypTensor``s support autograd similar to ``torch`` tensors. You can set
the ``requires_grad`` attribute on a ``CrypTensor`` to record forward
computations, and call ``backward()`` on a ``CrypTensor`` to perform
backpropagation.

.. code-block:: python

   # Create an CrypTensor that supports autograd
   x_enc = crypten.cryptensor(x_enc, requires_grad=True)
   y_enc = x_enc.mul(2).sum()
   y_enc.backward()

For an example of backpropagation using ``CrypTensor``, please
see `Tutorial 7
<https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb>`_.


File I/O Utilities
------------------

CrypTen provides utilities for loading ``CrypTensors`` from files and
saving ``CrypTensors`` to files.

.. automodule:: crypten
    :members: load, save
