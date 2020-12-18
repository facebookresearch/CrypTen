#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import pandas as pd


def get_aggregated_data(base_dir, subdirs):
    """Aggregate dataframe for model and func benchmarks assumining directory is structured as
        DATA_PATH
        |_2020-02-20
            |_subdir1
                |_func_benchmarks.csv
                |_model_benchmarks.csv
                |_func_benchmarks_cuda.csv (optional)
                |_model_benchmarks_cuda.csv (optional)
            |_subdir2
                ...
    Args:
        base_dir (pathlib.path): path containing month subdirectories
        subdirs (list): a list of all subdirectories to aggreagate dataframes from
    Returns: tuple of pd.DataFrames containing func and model benchmarks with dates
    """
    available_dates = get_available_dates(base_dir)
    func_df, model_df = pd.DataFrame(), pd.DataFrame()

    for subdir in subdirs:
        func_df_cpu, model_df_cpu = read_subdir(base_dir, available_dates, subdir)
        func_df_gpu, model_df_gpu = read_subdir(
            base_dir, available_dates, subdir, cuda=True
        )
        tmp_func_df = pd.concat([func_df_cpu, func_df_gpu])
        tmp_model_df = pd.concat([model_df_cpu, model_df_gpu])

        tmp_func_df["mode"] = subdir
        tmp_model_df["mode"] = subdir

        func_df = func_df.append(tmp_func_df)
        model_df = model_df.append(tmp_model_df)

    return func_df, model_df


def load_df(path, cuda=False):
    """Load dataframe for model and func benchmarks assumining directory is structured as
      path
        |_func_benchmarks.csv
        |_model_benchmarks.csv
        |_func_benchmarks_cuda.csv (optional)
        |_model_benchmarks_cuda.csv (optional)
    Args:
        path (str): path containing model and func benchmarks
        cuda (bool) : if set to true, read the corresponding func and model benchmarks for cuda
    Returns: tuple of pd.DataFrames containing func and model benchmarks with dates
    """
    postfix = "_cuda" if cuda else ""
    func_path = os.path.join(path, f"func_benchmarks{postfix}.csv")
    model_path = os.path.join(path, f"model_benchmarks{postfix}.csv")

    func_df, model_df = pd.DataFrame(), pd.DataFrame()
    if os.path.exists(func_path):
        func_df = pd.read_csv(func_path)
    if os.path.exists(model_path):
        model_df = pd.read_csv(model_path)

    return func_df, model_df


def read_subdir(base_dir, dates, subdir="", cuda=False):
    """Builds dataframe for model and func benchmarks assuming directory is structured as
     DATA_PATH
        |_2020-02-20
            |_subdir
                |_func_benchmarks.csv
                |_model_benchmarks.csv
                |_func_benchmarks_cuda.csv (optional)
                |_model_benchmarks_cuda.csv (optional)
    Args:
        base_dir (pathlib.path): path containing month subdirectories
        dates (list of str): containing dates / subdirectories available
        subdir (str) : string indicating the name of the sub directory to read enchmarks from
        cuda (bool) : if set to true, read the corresponding func and model benchmarks for cuda
    Returns: tuple of pd.DataFrames containing func and model benchmarks with dates
    """
    func_df, model_df = pd.DataFrame(), pd.DataFrame()
    device = "gpu" if cuda else "cpu"

    for date in dates:
        path = os.path.join(base_dir, date, subdir)

        tmp_func_df, tmp_model_df = load_df(path, cuda=cuda)
        set_metadata(tmp_func_df, date, device)
        set_metadata(tmp_model_df, date, device)

        func_df = func_df.append(tmp_func_df)
        model_df = model_df.append(tmp_model_df)

    if not func_df.empty:
        func_df = compute_runtime_gap(func_df)
        func_df = add_error_bars(func_df)
    return func_df, model_df


def get_available_dates(data_dir):
    """Returns list of available dates in DATA_PATH directory"""
    available_dates = []

    for sub_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, sub_dir)):
            available_dates.append(sub_dir)

    return available_dates


def set_metadata(df, date, device):
    """Set the device and date attribute for the dataframe"""
    df["date"] = date
    df["device"] = device


def compute_runtime_gap(func_df):
    """Computes runtime gap between CrypTen and Plain Text"""
    func_df["runtime gap"] = func_df["runtime crypten"] / func_df["runtime"]
    func_df["runtime gap Q1"] = func_df["runtime crypten Q1"] / func_df["runtime"]
    func_df["runtime gap Q3"] = func_df["runtime crypten Q3"] / func_df["runtime"]
    return func_df


def add_error_bars(func_df):
    """Adds error bars for plotting based on Q1 and Q3"""
    columns = ["runtime crypten", "runtime gap"]
    for col in columns:
        func_df = calc_error_bar(func_df, col)
    return func_df


def calc_error_bar(df, column_name):
    """Adds error plus and minus for plotting"""
    error_plus = df[column_name + " Q3"] - df[column_name]
    error_minus = df[column_name] - df[column_name + " Q1"]
    df[column_name + " error plus"] = error_plus
    df[column_name + " error minus"] = error_minus
    return df
