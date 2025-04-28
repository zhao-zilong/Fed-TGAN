import json
import pickle5 as pickle
from glob import glob

import numpy as np
import pandas as pd
from dtds.data.constants import CATEGORICAL, ORDINAL
from dtds.data.utils.file_generator import FileGenerator


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    if "npz" in filename:
        return loader(filename, allow_pickle=True)
    elif "pkl" in filename:
        with open(filename, "rb") as f:
            return loader(f)
    else:
        return loader(filename)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata["columns"]):
        if column["type"] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column["type"] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_datapath(path, benchmark=False):
    data = _load_file(glob(f"{path}/*.npz")[0], np.load)
    meta = _load_file(glob(f"{path}/*.json")[0], _load_json)
    # le = _load_file(glob(f"{path}/*.pickle")[0], pickle.load)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data["train"]
    if benchmark:
        return train, data["test"], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns 


def prepare_data(filename, selected_variables, categorical_list, nonnegative_list, date_dic, target_column, problem_type):

    df = pd.read_csv(filename)
    file_name = filename.split('/')[-1].split('.')[0]
    variables = df.columns.tolist()
    TARGET_COLUMN = "none" if target_column == "none" else target_column
    synthesizer_name = 'CTGANSynthesizer'
    file_generator = FileGenerator(
        dataframe=df[selected_variables],
        file_name=file_name,
        problem_type=problem_type,
        target_col="" if TARGET_COLUMN == "none" else TARGET_COLUMN,
        categorical_list=categorical_list,
        non_negative_columns=nonnegative_list,
        date_columns=date_dic,
        synthesizer_used=synthesizer_name
    )
    # Generate json and npz files and return encoded labels dictionary.
    meta_file = file_generator.generate_meta_data()
    return meta_file

def encode_data_with_meta_labelencoder(filename, meta_file, label_encoder, selected_variables,
categorical_list, nonnegative_list, date_dic, target_column, problem_type, timestamp):
    df = pd.read_csv(filename)
    file_name = filename.split('/')[-1].split('.')[0]
    variables = df.columns.tolist()
    TARGET_COLUMN = "none" if target_column == "none" else target_column
    synthesizer_name = 'CTGANSynthesizer'
    file_generator = FileGenerator(
        dataframe=df[selected_variables],
        file_name=file_name,
        problem_type=problem_type,
        target_col="" if TARGET_COLUMN == "none" else TARGET_COLUMN,
        categorical_list=categorical_list,
        non_negative_columns=nonnegative_list,
        date_columns=date_dic,
        synthesizer_used=synthesizer_name
    )
    file_generator.generate_data(meta_file, label_encoder, timestamp)
    return load_datapath(file_generator.output_path)

def load_dataset(name, benchmark=False):
    print(f"loading: {glob(f'data/processed/{name}*/*.*')[0].split('.')[0]}")
    data = _load_file(glob(f"data/processed/{name}*/*.npz")[0], np.load)
    meta = _load_file(glob(f"data/processed/{name}*/*.json")[0], _load_json)
    le = _load_file(glob(f"data/processed/{name}*/*.pkl")[0], pickle.load)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data["train"]
    if benchmark:
        return train, data["test"], meta, categorical_columns, ordinal_columns, le

    return train, categorical_columns, ordinal_columns, le
