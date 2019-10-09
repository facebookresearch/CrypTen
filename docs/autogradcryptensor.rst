AutogradCrypTensor
==================

``AutogradCrypTensors`` allow CrypTensors to store gradients and thus
enable backpropagation. ``AutogradCrypTensors`` can be created from
``CrypTensors`` and are only needed when backpropagation needs to be
performed. For all other operations on encrypted tensors,
``CrypTensors`` are sufficient.

To create an ``AutogradCrypTensor``,

.. code-block:: python

   # Create a CrypTensor
   x_enc = crypten.cryptensor(x)
   
   # Create an AutogradCrypTensor from the CrypTensor
   x_enc_auto = AutogradCrypTensor(x_enc)

For an example of how to carry out backpropagation
using ``AutogradCrypTensor``, please see Tutorial 7 
(i.e., tutorials/Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb)

.. autoclass:: crypten.autograd_cryptensor.AutogradCrypTensor 
   :members: backward, tensor    

    
    
