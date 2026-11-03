"""
data_cleaning.py
----------------
Loads and cleans labeled CSV data (synthetic or real).

Input columns:
- table_name, column_name, data_type (features)
- pii_class (label)

Cleaning pipeline (each step logged):
1. Normalize CSV headers (lowercase, aliases)
2. Strip whitespace from all columns
3. Drop exact duplicate rows
4. Drop rows with empty column_name or label
5. Validate labels against taxonomy
6. Drop near-duplicates (same after normalization)
7. Drop conflicting rows (same table/column/type with different labels)

Conflicting rows indicate label noise and are removed entirely.
"""
import os

import pandas as pd

from config import (
    COL_DATA_TYPE, COL_LABEL, COL_NAME, COL_TABLE_NAME,
)
from taxonomy import LABELS


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the CSV and normalize headers (lower-case, spaces -> underscores).

    Required columns (after normalization):
    - column_name (required)
    - pii_class (required)
    - table_name (optional, defaults to empty)
    - data_type (optional, defaults to empty)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input CSV not found at {path}. Run `python src/synthetic_data.py` to generate one, "
            "or pass --raw-csv."
        )
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    missing = {COL_NAME, COL_LABEL} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}.\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected columns: {COL_TABLE_NAME}, {COL_NAME}, {COL_DATA_TYPE}, {COL_LABEL}"
        )
    for optional_col in (COL_TABLE_NAME, COL_DATA_TYPE):
        if optional_col not in df.columns:
            print(f"NOTE: optional column '{optional_col}' not found; treating as empty.")
            df[optional_col] = ""
    return df[[COL_TABLE_NAME, COL_NAME, COL_DATA_TYPE, COL_LABEL]].copy()


def _strip_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    for col in [COL_TABLE_NAME, COL_NAME, COL_DATA_TYPE, COL_LABEL]:
        df[col] = _strip_series(df[col])
    df[COL_LABEL] = df[COL_LABEL].str.lower()

    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n - len(df)} exact duplicate row(s).")

    n = len(df)
    df = df[(df[COL_LABEL].str.len() > 0) & (df[COL_NAME].str.len() > 0)].reset_index(drop=True)
    print(f"Dropped {n - len(df)} row(s) with an empty column_name or label.")

    n = len(df)
    unknown = sorted(set(df[COL_LABEL]) - set(LABELS))
    if unknown:
        print(f"WARNING: labels not in taxonomy will be dropped: {unknown}")
    df = df[df[COL_LABEL].isin(LABELS)].reset_index(drop=True)
    print(f"Dropped {n - len(df)} row(s) with an out-of-taxonomy label.")

    key = (df[COL_TABLE_NAME].str.lower() + "|" + df[COL_NAME].str.lower()
           + "|" + df[COL_DATA_TYPE].str.lower())
    full_key = key + "|" + df[COL_LABEL]
    n = len(df)
    df = df.loc[~full_key.duplicated()].reset_index(drop=True)
    print(f"Dropped {n - len(df)} near-duplicate row(s).")

    key = (df[COL_TABLE_NAME].str.lower() + "|" + df[COL_NAME].str.lower()
           + "|" + df[COL_DATA_TYPE].str.lower())
    conflicting = df.groupby(key)[COL_LABEL].transform("nunique") > 1
    if conflicting.any():
        print(f"WARNING: {int(conflicting.sum())} row(s) share identical inputs but have different labels (label noise); dropping.")
    df = df[~conflicting].reset_index(drop=True)

    print(f"Rows remaining after cleaning: {len(df)} (started with {n_start}).")
    return df
