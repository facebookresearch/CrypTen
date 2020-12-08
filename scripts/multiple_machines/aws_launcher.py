#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This file is a tool to run MPC distributed training over AWS.

To run distributed training, first multiple AWS instances needs to be created
with a public AMI "Deep Learning AMI (Ubuntu) Version 24.0":

$ aws ec2 run-instances \
    --image-id ami-0ddba16a97b1dcda5 \
    --count 2 \
    --instance-type t2.micro \
    --key-name fair-$USER \
    --tag-specifications "ResourceType=instance,Tags=[{Key=fair-user,Value=$USER}]"

Two EC2 instances will be created by the command line shown above. Assume
the ids of the two instances created are i-068681e808235a851 and
i-0d7ebacfe1e3f28eb. Next, pytorch and crypten must be properly installed
on every instance.

Then the following command lines can run the mpc_linear_svm example on the two
EC2 instances created above:

$ python3 crypten/scripts/aws_launcher.py \
    --SSH_keys=/home/$USER/.aws/fair-$USER.pem \
    --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
    --aux_files=crypten/examples/mpc_linear_svm/mpc_linear_svm.py \
    crypten/examples/mpc_linear_svm/launcher.py \
        --features 50 \
        --examples 100 \
        --epochs 50 \
        --lr 0.5 \
        --skip_plaintext


If you want to train with AWS instances located at multiple regions, then you would need
to provide ssh_key_file for each instance:

$ python3 crypten/scripts/aws_launcher.py \
    --regions=us-east-1,us-west-1 \
    --SSH_keys=/home/$USER/.aws/east.pem,/home/$USER/.aws/west.pem  \
    --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
    --aux_files=crypten/examples/mpc_linear_svm/mpc_linear_svm.py \
    crypten/examples/mpc_linear_svm/launcher.py \
        --features 50 \
        --examples 100 \
        --epochs 50 \
        --lr 0.5 \
        --skip_plaintext

"""

import configparser
import os
import sys
import time
import warnings
from argparse import REMAINDER, ArgumentParser
from pathlib import Path

import boto3
import paramiko

import common

def get_instances(ec2, instance_ids):
    instances = list(
        ec2.instances.filter(Filters=[{"Name": "instance-id", "Values": instance_ids}])
    )
    return instances


def main():
    parser = get_parser()
    args = parser.parse_args()

    cf = configparser.ConfigParser()
    cf.read(args.credentials)

    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
    )

    regions = args.regions.split(",")
    instance_ids = args.instances.split(",")
    ssh_key_files = args.ssh_key_file.split(",")

    instances = []

    # AWS Specific
    if len(regions) > 1:
        print("Multiple regions detected")

        assert len(instance_ids) == len(
            ssh_key_files
        ), "{} instance ids are provided, but {} SSH keys found.".format(
            len(instance_ids), len(ssh_key_files)
        )

        assert len(instance_ids) == len(
            regions
        ), "{} instance ids are provided, but {} regions found.".format(
            len(instance_ids), len(regions)
        )

        for i, region in enumerate(regions):
            session = boto3.session.Session(
                aws_access_key_id=cf["default"]["aws_access_key_id"],
                aws_secret_access_key=cf["default"]["aws_secret_access_key"],
                region_name=region,
            )
            ec2 = session.resource("ec2")

            instance = get_instances(ec2, [instance_ids[i]])
            instances += instance
    else:
        session = boto3.session.Session(
            aws_access_key_id=cf["default"]["aws_access_key_id"],
            aws_secret_access_key=cf["default"]["aws_secret_access_key"],
            region_name=regions[0],
        )
        ec2 = session.resource("ec2")
        instances = get_instances(ec2, instance_ids)

        assert (
            len(ssh_key_files) == 1
        ), "1 region is detected, but {} SSH keys found.".format(len(ssh_key_files))

        ssh_key_files = [ssh_key_files[0] for _ in range(len(instances))]

    assert len(instance_ids) == len(
        instances
    ), "{} instance ids are provided, but {} found.".format(
        len(instance_ids), len(instances)
    )

    # Only print the public IP addresses of the instances.
    # Then do nothing else and return.
    if args.only_show_instance_ips:
        for instance in instances:
            print(instance.public_ip_address)
        return

    world_size = len(instances)
    print(f"Running world size {world_size} with instances: {instances}")
    master_instance = instances[0]

    # Key: instance id; value: paramiko.SSHClient object.
    client_dict = {}
    for ssh_key_file, instance in zip(ssh_key_files, instances):
        client = common.connect_to_machine(
            instance.public_ip_address, ssh_key_file, args.ssh_user, args.http_proxy
        )
        client_dict[instance.public_ip_address] = client

    remote_dir, script_basename = common.upload_files_to_machines(client_dict, ars.file_paths, args.script)

    environment = {
        "WORLD_SIZE": str(world_size),
        "RENDEZVOUS": "env://",
        "MASTER_ADDR": master_instance.private_ip_address,
        "MASTER_PORT": str(args.master_port),
    }

    common.run_script_parallel(environment)

    common.cleanup(client_dict, remote_dir)


def get_parser():
    parser = common.get_parser()

    """ Add AWS specific arguments """
    parser.add_argument(
        "--only_show_instance_ips",
        action="store_true",
        default=False,
        help="Only show public IPs of the given instances."
        "No other actions will be done",
    )

    parser.add_argument("--regions", type=str, default="us-west-2", help="AWS Region")

    parser.add_argument(
        "--instances",
        type=str,
        required=True,
        help="The comma-separated ids of AWS instances",
    )

    return parser


if __name__ == "__main__":
    main()
