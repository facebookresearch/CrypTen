#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains helper function to run MPC over multiple machines

"""

import concurrent.futures
import sys
import time
import uuid
from argparse import REMAINDER, ArgumentParser
from pathlib import Path

import paramiko


def run_command(machine_ip, client, cmd, environment=None, inputs=None):
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
        printer.submit(read_lines, stdout, sys.stdout, f"[{machine_ip} STDOUT] ")
        printer.submit(read_lines, stderr, sys.stderr, f"[{machine_ip} STDERR] ")


def upload_files_to_machines(client_dict, aux_files, script):
    path_script = Path(script)

    assert path_script.exists(), f"File `{script}` does not exist"
    file_paths = aux_files.split(",") if aux_files else []

    for local_path in file_paths:
        assert Path(local_path).exists(), f"File `{local_path}` does not exist"

    remote_dir = Path(f"tmp-dir-{uuid.uuid1()}")
    script_basename = path_script.name
    remote_script = remote_dir / script_basename

    print(f"Remote path {remote_dir}")

    def upload_file(machine_ip, client, localpath, remotepath):
        ftp_client = client.open_sftp()
        print(f"Uploading `{localpath}` to {machine_ip} as {remotepath}...")
        ftp_client.put(localpath, str(remotepath), confirm=True)
        ftp_client.close()
        print(f"`{localpath}` uploaded to {machine_ip}")

    # Upload files to all machines concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as uploaders:
        for machine_ip, client in client_dict.items():
            run_command(machine_ip, client, f"mkdir {remote_dir}")
            uploaders.submit(
                upload_file, machine_ip, client, script, remote_script
            )
            for local_path in file_paths:
                local_path = Path(local_path)
                uploaders.submit(
                    upload_file,
                    machine_ip,
                    client,
                    local_path,
                    remote_dir / local_path.name,
                )

    for machine_ip, client in client_dict.items():
        run_command(machine_ip, client, f"chmod +x {remote_script}")
        run_command(machine_ip, client, f"ls -al {remote_dir}")

    return remote_dir, script_basename


def run_script_parallel(
        environment,
        client_dict,
        remote_dir,
        script_basename,
        script_args=None,
        prepare_cmd=None):

    world_size = len(client_dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as executor:
        for rank, (machine_ip, client) in enumerate(client_dict.items()):
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
            prep_cmd = f"{prepare_cmd}; " if prepare_cmd else ""
            if rank == 0:
                environment_cmd += ";export GLOO_SOCKET_IFNAME=wlp59s0"
            else:
                environment_cmd += ";export GLOO_SOCKET_IFNAME=wlp2s0"
            cmd = "{}; {} {} {} {}".format(
                environment_cmd,
                f"cd {remote_dir} ;",
                prep_cmd,
                f"./{script_basename}",
                " ".join(script_args),
            )
            print(f"Run command: {cmd}")
            executor.submit(run_command, machine_ip, client, cmd, environment)


def connect_to_machine(ip_address, keypath, username, http_proxy=None, retries=20):
    print(f"Connecting to {ip_address}...")

    proxy = None
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

    while retries > 0:
        try:
            client.connect(
                ip_address,
                username=username,
                key_filename=keypath,
                timeout=10,
                sock=proxy,
            )
            print(f"Connected to {ip_address}")
            break
        except Exception as e:
            print(f"Exception: {e} Retrying...")
            retries -= 1
            time.sleep(10)
    return client


def cleanup(client_dict, remote_dir):
    for machine_ip, client in client_dict.items():
        run_command(machine_ip, client, f"rm -rf {remote_dir}")
        client.close()


def get_parser():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed launcher "
        "helper utilty that will spawn up "
        "parties for MPC scripts on different machines"
    )

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="The port used by master instance for MPC",
    )

    parser.add_argument(
        "--ssh_key_file",
        type=str,
        required=True,
        help="Path to the RSA private key file used for authentication",
    )

    parser.add_argument(
        "--ssh_user",
        type=str,
        default="ubuntu",
        help="The username to ssh to the other machines"
    )

    parser.add_argument(
        "--http_proxy",
        type=str,
        default=None,
        help="If not none, use the http proxy specified "
        "(e.g., fwdproxy:8080) to ssh to machines",
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
        "script",
        type=str,
        help="The full path to the single "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the script",
    )

    # the rest of the arguments are passed to the script
    parser.add_argument("script_args", nargs=REMAINDER)
    return parser
