#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to run historical benchmarks.

- writes monthly data to 'dash_app/data/`
    - example: 'dash_app/data/2019-10-26/func_benchmarks.csv'
    - example: 'dash_app/data/2019-10-26/model_benchmarks.csv'
- overwrite option
- script requires ability to 'git clone'

To run:
python run_historical_benchmarks.py

# overwrite existing data directories
python run_historical_benchmarks.py --overwrite True
"""

import argparse
import datetime
import os
import shutil
import subprocess

from dateutil.relativedelta import relativedelta


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Run Historical Benchmarks")
    parser.add_argument(
        "--overwrite",
        required=False,
        default=False,
        action="store_true",
        help="overwrite existing data directories",
    )
    parser.add_argument(
        "--cuda-toolkit-version",
        required=False,
        default="10.1",
        help="build pytorch with the corresponding version of cuda-toolkit",
    )
    args = parser.parse_args()
    return args


def get_dates(day=26):
    """Generate dates to run benchmarks

    Returns: list of strings in year-month-day format.
        Example: ["2020-01-26", "2019-12-26"]
    """
    dates = []
    today = datetime.date.today()
    end = datetime.date(2019, 10, day)
    one_month = relativedelta(months=+1)

    if today.day >= 26:
        start = datetime.date(today.year, today.month, day)
    else:
        start = datetime.date(today.year, today.month, day) - one_month

    while start >= end:
        dates.append(start.strftime("%Y-%m-%d"))
        start -= one_month

    return dates


args = parse_args()
overwrite = args.overwrite
cuda_version = "".join(args.cuda_toolkit_version.split("."))
dates = get_dates()
PATH = os.getcwd()


# clone
subprocess.call(
    "cd /tmp && git clone https://github.com/facebookresearch/CrypTen.git", shell=True
)

# create venv
subprocess.call("cd /tmp && python3 -m venv .venv", shell=True)
venv = "cd /tmp && . .venv/bin/activate && "

# install PyTorch
subprocess.call(
    f"{venv} pip3 install onnx==1.6.0 tensorboard pandas sklearn", shell=True
)
stable_url = "https://download.pytorch.org/whl/torch_stable.html"
pip_torch = f"pip install torch==1.5.1+cu{cuda_version} torchvision==0.6.1+cu{cuda_version} -f https://download.pytorch.org/whl/torch_stable.html"
subprocess.call(f"{venv} {pip_torch} -f {stable_url}", shell=True)


modes = {"1pc": "", "2pc": "--world-size=2"}

for date in dates:
    path_exists = os.path.exists(f"dash_app/data/{date}/func_benchmarks.csv")
    if not overwrite and path_exists:
        continue
    # checkout closest version before date
    subprocess.call(
        f"cd /tmp/CrypTen && "
        + f"git checkout `git rev-list -n 1 --before='{date} 01:01' master`",
        shell=True,
    )
    for mode, arg in modes.items():
        subprocess.call(venv + "pip3 install CrypTen/.", shell=True)
        subprocess.call(f"echo Generating {date} Benchmarks for {mode}", shell=True)
        path = os.path.join(PATH, f"dash_app/data/{date}", mode)
        subprocess.call(f"mkdir -p {path}", shell=True)
        subprocess.call(
            venv + f"cd {PATH} && python3 benchmark.py -p '{path}' {arg}", shell=True
        )
        subprocess.call(
            venv + f"cd {PATH} && python3 benchmark.py -p '{path}' -d 'cuda' {arg}",
            shell=True,
        )

# clean up
shutil.rmtree("/tmp/.venv", ignore_errors=True)
shutil.rmtree("/tmp/CrypTen", ignore_errors=True)
