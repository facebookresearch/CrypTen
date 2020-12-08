#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This file is a tool to run MPC over multiple machines (given a
set of IP addresses)

python3 crypten/scripts/aws_launcher.py \
    --SSH_keys=/home/$USER/ \
    --instances=192.168.0.1,192.168.0.2 \
    --aux_files=crypten/examples/mpc_linear_svm/mpc_linear_svm.py \
    crypten/examples/mpc_linear_svm/launcher.py \
        --features 50 \
        --examples 100 \
        --epochs 50 \
        --lr 0.5 \
        --skip_plaintext
"""

import concurrent.futures
import common
import configparser
import warnings

def main():
    parser = common.get_parser()
    args = parser.parse_args()

    cf = configparser.ConfigParser()
    cf.read(args.credentials)

    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
    )

    ip_addresses = args.ip_addresses.split(",")
    ssh_key_files = args.ssh_key_file.split(",")

    if len(ssh_key_files) == 1:
        ssh_key_files = [ssh_key_files[0] for _ in range(len(ip_addresses))]

    world_size = len(ip_addresses)
    print(f"Running world size {world_size} with ip_addresses: {ip_addresses}")
    master_ip_address = ip_addresses[0]

    # Key: instance id; value: paramiko.SSHClient object.
    client_dict = {}
    for ssh_key_file, ip_address in zip(ssh_key_files, ip_addresses):
        client = common.connect_to_machine(
            ip_address, ssh_key_file, args.ssh_user, args.http_proxy
        )
        client_dict[ip_address] = client

    remote_dir, script_basename = common.upload_files_to_machines(client_dict, args.aux_files, args.script)

    environment = {
        "WORLD_SIZE": str(world_size),
        "RENDEZVOUS": "env://",
        "MASTER_ADDR": master_ip_address,
        "MASTER_PORT": str(args.master_port),
    }

    kwargs = {
        "environment": environment,
        "client_dict": client_dict,
        "remote_dir": remote_dir,
        "script_basename": script_basename,
        "script_args": args.script_args,
        "prepare_cmd": args.prepare_cmd
    }
    common.run_script_parallel(**kwargs)

    common.cleanup(client_dict, remote_dir)


if __name__ == "__main__":
    main()
