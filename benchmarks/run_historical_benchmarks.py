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


def parse_overwrite():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Run Historical Benchmarks")
    parser.add_argument(
        "--overwrite",
        type=bool,
        required=False,
        default=False,
        help="overwrite existing data directories",
    )
    args = parser.parse_args()
    return args.overwrite


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


overwrite = parse_overwrite()
dates = get_dates()
PATH = os.getcwd()


# clone
subprocess.call(
    "cd /tmp && git clone https://github.com/facebookresearch/CrypTen.git", shell=True
)

# create venv
subprocess.call("cd /tmp && python -m venv .venv", shell=True)
venv = "cd /tmp && source .venv/bin/activate && "

# install PyTorch
subprocess.call(
    f"{venv} pip install onnx==1.6.0 tensorboard pandas sklearn", shell=True
)
nightly = "dev20200220"
nightly_url = "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
pip_torch = f"pip install torch==1.5.0.{nightly} torchvision==0.6.0.{nightly}"
subprocess.call(f"{venv} {pip_torch} -f {nightly_url}", shell=True)


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
    subprocess.call(venv + "pip install CrypTen/.", shell=True)
    subprocess.call(f"echo Generating {date} Benchmarks", shell=True)
    subprocess.call(f"mkdir -p dash_app/data/{date}", shell=True)
    path = os.path.join(PATH, f"dash_app/data/{date}")
    subprocess.call(venv + f"cd {PATH} && python benchmark.py -p '{path}'", shell=True)


# clean up
shutil.rmtree("/tmp/.venv", ignore_errors=True)
shutil.rmtree("/tmp/CrypTen", ignore_errors=True)
