MPCTensor
=========

An ``MPCTensor`` is a ``CrypTensor`` encrypted using the secure MPC protocol.
In order to support the mathematical operations required by the ``MPCTensor``,
CrypTen implements two kinds of secret-sharing protocols:
arithmetic secret-sharing and binary secret-sharing.
Arithmetic secret sharing forms the basis for most of the mathematical operations implemented by ``MPCTensor``.
Similarly, binary secret-sharing allows for the evaluation of logical expressions.

.. automodule:: crypten.mpc.mpc
    :members:
