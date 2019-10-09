Launch on AWS
=============

CrypTen also provides a script ``aws_launcher`` (see ``scripts/aws_launcher.py``) that will allow you
to compute on encrypted data across multiple AWS instances.

For example, if Alice has a classifier on one AWS
instance and Bob has data on another AWS instance,
``aws_launcher`` will allow Alice and Bob to classify the data
without revealing their respective private information.

The steps to follow are:

1. First, create multiple AWS instances with public AMI "Deep Learning AMI (Ubuntu) Version 24.0", and record the instance IDs.
2. Install PyTorch, CrypTen and dependencies of the program to be run on all AWS instances.
3. Run ``aws_launcher.py`` on your local machine, as we explain below.

The results are left on the AWS instances. Log messages will be printed on your local machine by launcher script.

To launch the `mpc_linear_svm` example,

.. code-block:: bash

    python3 [PATH_TO_CRYPTEN]/CrypTen/aws_launcher.py \
    --ssh_key_file [SSH_KEY_FILE] --instances=[AWS_INSTANCE1, AWS_INSTANCE2...] \
    --region [AWS_REGION] \
    --ssh_user [AWS_USERNAME] \
    --aux_files=[PATH_TO_CRYPTEN]/CrypTen/examples/mpc_linear_svm/mpc_linear_svm.py [PATH_TO_CRYPTEN]/CrypTen/examples/mpc_linear_svm/launcher.py \
    --features 50 \
    --examples 100 \
    --epochs 50 \
    --lr 0.5 \
    --skip_plaintext

Note you need to replace arguments with your own AWS instances, usernames, and ``.ssh`` keys.
