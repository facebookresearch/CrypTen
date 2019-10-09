nn
==

``crypten.nn`` provides modules for defining and training neural
networks similar to ``torch.nn``.

From PyTorch to CrypTen
-----------------------

The simplest way to create a CrypTen network is to start with a
PyTorch network, and use the ``from_pytorch`` function to convert it
to a CrypTen network. This is particularly useful for pre-trained
PyTorch networks that need to be encrypted before use. 

.. automodule:: crypten.nn
    :members: from_pytorch


.. note::

    In addition to the PyTorch network, the `from_pytorch` function
    also requires a dummy input of the shape of the model's input.
    The dummy input simply needs to be a `torch` tensor of the same
    shape; the values inside the tensor do not matter. For a complete
    example of how to use ``from_pytorch`` function, please see
    `Tutorial 4 <https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb>`_.

Modules
-------

``crypten.nn`` also provides several modules and containers
for directly building neural network. Here is an example of how
to use these objects to build a CrypTen network: 

.. code-block:: python
       
       model = crypten.nn.Sequential(
            [
                crypten.nn.Linear(num_inputs, num_intermediate),
                crypten.nn.ReLU(),
                crypten.nn.Linear(num_intermediate, num_outputs),
            ]
        )

Generic ``Modules``
+++++++++++++++++++++++++++

.. autoclass:: crypten.nn.module.Module
    :members: train, eval, encrypt, decrypt, forward, zero_grad, modules, named_modules, named_parameters, update_parameters

.. autoclass:: crypten.nn.module.Container
    :members:

.. autoclass:: crypten.nn.module.Graph
    :members:

.. autoclass:: crypten.nn.module.Sequential
    :members:


``Modules`` for Encrypted Layers
++++++++++++++++++++++++++++++++++++++++

.. autoclass:: crypten.nn.module.Constant
   :members: forward

.. autoclass:: crypten.nn.module.Add
   :members: forward	

.. autoclass:: crypten.nn.module.Sub
   :members: forward	

.. autoclass:: crypten.nn.module.Squeeze
   :members: forward

.. autoclass:: crypten.nn.module.Unsqueeze
   :members: forward   

.. autoclass:: crypten.nn.module.Flatten
   :members: forward   

.. autoclass:: crypten.nn.module.Shape
   :members: forward

.. autoclass:: crypten.nn.module.Concat
   :members: forward

.. autoclass:: crypten.nn.module.Gather
   :members: forward   

.. autoclass:: crypten.nn.module.Reshape
   :members: forward

.. autoclass:: crypten.nn.module.ConstantPad1d
   :members: forward

.. autoclass:: crypten.nn.module.ConstantPad2d
   :members: forward	   

.. autoclass:: crypten.nn.module.ConstantPad3d
   :members: forward

.. autoclass:: crypten.nn.module.Linear
   :members: forward

.. autoclass:: crypten.nn.module.Conv2d
   :members: forward	

.. autoclass:: crypten.nn.module.ConstantPad1d
   :members: forward

.. autoclass:: crypten.nn.module.AvgPool2d
   :members: forward

.. autoclass:: crypten.nn.module.MaxPool2d
   :members: forward

.. autoclass:: crypten.nn.module.GlobalAveragePool
   :members: forward   

.. autoclass:: crypten.nn.module.BatchNorm1d
   :members: forward

.. autoclass:: crypten.nn.module.BatchNorm2d
   :members: forward

.. autoclass:: crypten.nn.module.BatchNorm3d
   :members: forward

Loss Functions
---------------
CrypTen also provides a number of encrypted loss functions similar
to `torch.nn`. 

.. automodule:: crypten.nn.loss
    :members:
