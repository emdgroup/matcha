"""Serialization utilities for loading and saving data in various formats.

Supports JSON, YAML, and pickle serialization with automatic directory
creation, as well as DataFrame parsing helpers for molecular data.
"""

import pickle
import json
import yaml
import os
import numpy as np
import pandas as pd


def _sanitize_for_yaml(obj):
    """Recursively convert numpy types to native Python types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path: str, object: object):
    """Saves the object in the desired path as a readable json"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(object, f, indent=4)


def save_yaml(path: str, object: object):
    """Saves the object in the desired path as a readable yaml"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            _sanitize_for_yaml(object), f, default_flow_style=False, sort_keys=False
        )


def save_pickle(path: str, object: object):
    """Saves the object in the desired path as a pickle"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(object, f)


def load_json(path: str):
    """Loads the json as a dictionary"""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_pickle(path: str):
    """Reloads the object from a pickle file"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_yaml(path: str):
    """Loads a yaml dict from file"""
    with open(path, "r") as meta_file:
        meta_data = yaml.safe_load(meta_file)
    return meta_data


def parse_df(
    input_df: pd.DataFrame,
    label_key: str | list,
    operator_key: str | None = None,
):
    """Parse a DataFrame containing molecular data into components.

    Extracts RDKit molecule objects, label arrays, and optional censor
    operator lists from a DataFrame that has an ``ROMol`` column.

    :param pandas.DataFrame input_df: Input DataFrame with an ``ROMol`` column
        and label/operator columns.
    :param label_key: Column name substring (or list of exact column names)
        used to identify label columns.
    :type label_key: str or list
    :param operator_key: Optional column name substring used to identify
        censor operator columns (e.g., ``"<"``, ``">"``). NaN values are
        replaced with ``"="``.
    :type operator_key: str or None
    :returns: A tuple of (molecules, labels, operators) where molecules is a
        list of RDKit Mol objects, labels is a float numpy array, and operators
        is a list of operator strings (or ``None`` if ``operator_key`` is None).
    :rtype: tuple
    """
    cols = list(input_df.columns)
    mols = input_df.ROMol.tolist()

    if isinstance(operator_key, str):
        operator_cols = [x for x in cols if operator_key in x]
        operator = [input_df[x].astype(str).tolist() for x in operator_cols]
        operator = [[s.replace("nan", "=") for s in x] for x in operator]
        if len(operator_cols) == 1:
            operator = operator[0]
        input_df = input_df.drop(operator_cols, axis=1)

    elif operator_key is None:
        operator = None

    if isinstance(label_key, str):
        label_key = [x for x in cols if label_key in x]
    y_cols = [x for x in cols if x in label_key]
    y_df = input_df[y_cols]
    y = y_df.astype(float).to_numpy()

    return mols, y, operator
