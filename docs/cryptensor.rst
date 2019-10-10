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

``CrypTensors`` provide various operations similar to ``torch`` tensors.

.. automodule:: crypten.cryptensor
    :members:

File I/O Utilities
------------------

CrypTen provides utilities for loading ``CrypTensors`` from files and
saving ``CrypTensors`` to files. 

.. automodule:: crypten
    :members: load, save
