MPCTensor
=========

An ``MPCTensor`` is a ``CrypTensor`` encrypted using the secure MPC protocol.
In order to support the mathematical operations required by the ``MPCTensor``,
CrypTen implements two kinds of secret-sharing protocols defined by ``ptype``:

* ``crypten.mpc.arithmetic`` for arithmetic secret-sharing
* ``crypten.mpc.binary`` for binary secret-sharing

Arithmetic secret sharing forms the basis for most of the mathematical
operations implemented by ``MPCTensor``.  Similarly, binary
secret-sharing allows for the evaluation of logical expressions.

We can use the ``ptype`` attribute to create a ``CrypTensor`` with the appropriate
secret-sharing protocol. For example:

.. code-block:: python

  # arithmetic secret-shared tensors
  x_enc = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)
  print("x_enc internal type:", x_enc.ptype)

  # binary secret-shared tensors
  y_enc = crypten.cryptensor([1, 2, 1], ptype=crypten.mpc.binary)
  print("y_enc internal type:", y_enc.ptype)


We also provide helpers to execute secure multi-party computations
in separate processes (see :ref:`communicator`).

For technical details see `Damgard et al. 2012`_ and `Beaver 1991`_ outlining the Beaver protocol
used in our implementation.

.. _Damgard et al. 2012: https://eprint.iacr.org/2011/535.pdf
.. _Beaver 1991: https://link.springer.com/chapter/10.1007/3-540-46766-1_34

For examples illustrating arithmetic and binary secret-sharing in
CrypTen, the ``ptype`` attribute, and the execution of secure
multi-party computations, please see `Tutorial 2
<https://github.com/facebookresearch/CrypTen/blob/master/tutorials/Tutorial_2_Inside_CrypTensors.ipynb>`_.

Tensor Operations
-----------------

.. automodule:: crypten.mpc.mpc
    :members:

.. _communicator:

Communicator
------------

To execute multi-party computations locally, we provide
a ``@mpc.run_multiprocess`` function decorator,
which we developed to execute CrypTen code from a single script.
CrypTen follows the standard MPI programming model: it runs a separate
process for each party, but each process runs an identical (complete) program.
Each process has a ``rank`` variable to identify itself.

For example, two-party arithmetic secret-sharing:

.. code-block:: python

  import crypten
  import crypten.communicator as comm

  @mpc.run_multiprocess(world_size=2)
  def examine_arithmetic_shares():
      x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.mpc.arithmetic)

      rank = comm.get().get_rank()
      print(f"Rank {rank}:\n {x_enc}")

  x = examine_arithmetic_shares()

.. autofunction:: crypten.mpc.context.run_multiprocess
.. autofunction:: crypten.communicator.Communicator.get_world_size
.. autofunction:: crypten.communicator.Communicator.get_rank
