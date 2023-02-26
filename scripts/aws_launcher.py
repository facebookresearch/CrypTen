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

import concurrent.futures
import configparser
import os
import sys
import time
import uuid
import warnings
from argparse import ArgumentParser, REMAINDER
from pathlib import Path

import boto3
import paramiko


def get_instances(ec2, instance_ids):
    instances = list(
        ec2.instances.filter(Filters=[{"Name": "instance-id", "Values": instance_ids}])
    )
    return instances


def connect_to_instance(instance, keypath, username, http_proxy=None):
    print(f"Connecting to {instance.id}...")

    ip_address = instance.public_ip_address
    if http_proxy:
        # paramiko.ProxyCommand does not do string substitution for %h %p,
        # so 'nc --proxy-type http --proxy fwdproxy:8080 %h %p' would not work!
        proxy = paramiko.ProxyCommand(
            f"nc --proxy-type http --proxy {http_proxy} {ip_address} {22}"
        )
        proxy.settimeout(300)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    retries = 20
    while retries > 0:
        try:
            client.connect(
                ip_address,
                username=username,
                key_filename=keypath,
                timeout=10,
                sock=proxy if http_proxy else None,
            )
            print(f"Connected to {instance.id}")
            break
        except Exception as e:
            print(f"Exception: {e} Retrying...")
            retries -= 1
            time.sleep(10)
    return client


def add_prefix_each_line(prefix, str):
    lines = [f"{prefix}{line}" for line in str.split("\n")]
    return "\n".join(lines)


def run_command(instance, client, cmd, environment=None, inputs=None):
    stdin, stdout, stderr = client.exec_command(
        cmd, get_pty=True, environment=environment
    )
    if inputs:
        for inp in inputs:
            stdin.write(inp)

    def read_lines(fin, fout, line_head):
        line = ""
        while not fin.channel.exit_status_ready():
            line += fin.read(1).decode("utf8")
            if line.endswith("\n"):
                print(f"{line_head}{line[:-1]}", file=fout)
                line = ""
        if line:
            # print what remains in line buffer, in case fout does not
            # end with '\n'
            print(f"{line_head}{line[:-1]}", file=fout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as printer:
        printer.submit(read_lines, stdout, sys.stdout, f"[{instance} STDOUT] ")
        printer.submit(read_lines, stderr, sys.stderr, f"[{instance} STDERR] ")


def upload_file(instance_id, client, localpath, remotepath):
    ftp_client = client.open_sftp()
    print(f"Uploading `{localpath}` to {instance_id}...")
    ftp_client.put(localpath, remotepath)
    ftp_client.close()
    print(f"`{localpath}` uploaded to {instance_id}.")


def main():
    args = parse_args()

    cf = configparser.ConfigParser()
    cf.read(args.credentials)

    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
    )

    regions = args.regions.split(",")
    instance_ids = args.instances.split(",")
    ssh_key_files = args.ssh_key_file.split(",")

    instances = []
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
    for i, instance in enumerate(instances):
        client = connect_to_instance(
            instance, ssh_key_files[i], args.ssh_user, args.http_proxy
        )
        client_dict[instance.id] = client

    assert os.path.exists(
        args.training_script
    ), f"File `{args.training_script}` does not exist"
    file_paths = args.aux_files.split(",") if args.aux_files else []
    for local_path in file_paths:
        assert os.path.exists(local_path), f"File `{local_path}` does not exist"

    remote_dir = f"aws-launcher-tmp-{uuid.uuid1()}"
    script_basename = os.path.basename(args.training_script)
    remote_script = os.path.join(remote_dir, script_basename)

    # Upload files to all instances concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as uploaders:
        for instance_id, client in client_dict.items():
            run_command(instance_id, client, f"mkdir -p {remote_dir}")
            uploaders.submit(
                upload_file, instance_id, client, args.training_script, remote_script
            )
            for local_path in file_paths:
                uploaders.submit(
                    upload_file,
                    instance_id,
                    client,
                    local_path,
                    os.path.join(remote_dir, os.path.basename(local_path)),
                )
    for instance_id, client in client_dict.items():
        run_command(instance_id, client, f"chmod +x {remote_script}")
        run_command(instance_id, client, f"ls -al {remote_dir}")

    environment = {
        "WORLD_SIZE": str(world_size),
        "RENDEZVOUS": "env://",
        "MASTER_ADDR": master_instance.private_ip_address,
        "MASTER_PORT": str(args.master_port),
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as executor:
        rank = 0
        for instance_id, client in client_dict.items():
            environment["RANK"] = str(rank)
            # TODO: Although paramiko.SSHClient.exec_command() can accept
            # an argument `environment`, it seems not to take effect in
            # practice. It might because "Servers may silently reject
            # some environment variables" according to paramiko document.
            # As a workaround, here all environment variables are explicitly
            # exported.
            environment_cmd = "; ".join(
                [f"export {key}={value}" for (key, value) in environment.items()]
            )
            prepare_cmd = f"{args.prepare_cmd}; " if args.prepare_cmd else ""
            cmd = "{}; {} {} {} {}".format(
                environment_cmd,
                f"cd {remote_dir} ;",
                prepare_cmd,
                f"./{script_basename}",
                " ".join(args.training_script_args),
            )
            print(f"Run command: {cmd}")
            executor.submit(run_command, instance_id, client, cmd, environment)
            rank += 1

    # Cleanup temp dir.
    for instance_id, client in client_dict.items():
        run_command(instance_id, client, f"rm -rf {remote_dir}")
        client.close()


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "parties for MPC scripts on AWS"
    )

    parser.add_argument(
        "--credentials",
        type=str,
        default=f"{Path.home()}/.aws/credentials",
        help="Credentials used to access AWS",
    )

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

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="The port used by master instance " "for distributed training",
    )

    parser.add_argument(
        "--ssh_key_file",
        type=str,
        required=True,
        help="Path to the RSA private key file " "used for instance authentication",
    )

    parser.add_argument(
        "--ssh_user",
        type=str,
        default="ubuntu",
        help="The username to ssh to AWS instance",
    )

    parser.add_argument(
        "--http_proxy",
        type=str,
        default=None,
        help="If not none, use the http proxy specified "
        "(e.g., fwdproxy:8080) to ssh to AWS instance",
    )

    parser.add_argument(
        "--aux_files",
        type=str,
        default=None,
        help="The comma-separated paths of additional files "
        " that need to be transferred to AWS instances. "
        "If more than one file needs to be transferred, "
        "the basename of any two files can not be the "
        "same.",
    )

    parser.add_argument(
        "--prepare_cmd",
        type=str,
        default="",
        help="The command to run before running distribute "
        "training for prepare purpose, e.g., setup "
        "environment, extract data files, etc.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single machine training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    main()
