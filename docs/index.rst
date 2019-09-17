.. CrypTen documentation master file, created by
   sphinx-quickstart on Thu Sep 12 20:49:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CrypTen Documentation
=====================
CrypTen is a Privacy Preserving Machine Learning framework written using `PyTorch
<http://pytorch.org/>`_ that allows researchers and developers to
train models using encrypted data. CrypTen currently supports
`Secure multi-party computation
<https://en.wikipedia.org/wiki/Secure_multi-party_computation>`_ as its encryption
mechanism.

Installation on Linux and Mac
=============================

We recommend installing CrypTen in its own ``conda`` environment. Please install
``Anaconda Python 3.7`` before doing the following steps

.. code-block:: bash

    $ conda create -n crypten-env python=3.7
    $ conda activate crypten-env
    $ conda install pytorch torchvision cpuonly -c pytorch-nightly 
    $ git clone git@github.com:facebookresearch/CrypTen.git
    $ cd CrypTen; python3 setup.py install

To check if your installation is working, you can run the unit tests as follows

.. code-block:: bash

    $ python3 -m unittest discover test

We do not support Windows yet.

Examples
========

To run the examples in the ``examples`` directory, you additionally need to do
the following

.. code-block:: bash

    $ pip install -r requirements.examples.txt

For any example, the following should show the available command line options

.. code-block:: bash

    $ python3 examples/<example>/launcher.py --help

The linear SVM example, generates random data and trains a SVM classifier.
On a single node, you can run it as follows

.. code-block:: bash

    $ python3 examples/mpc_linear_svm/launcher.py --multiprocess


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
