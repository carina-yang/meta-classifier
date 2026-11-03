"""
label_encoding.py
-----------------
Encodes string labels to stable integer IDs based on taxonomy order.

Uses fixed taxonomy.LABELS order, ensuring:
- Stable IDs across runs and datasets
- Absent classes don't shift other class IDs
- Integer encoding: 0=person_name, 1=email, ... 9=non_pii

Returns:
- DataFrame with integer 'label' column
- Label mapping dictionary {tag: id}
- Index-to-label list for decoding predictions
"""
from typing import Dict, List, Tuple

import pandas as pd

from config import COL_LABEL, LABEL_ID_COLUMN
from taxonomy import LABELS


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Returns (df with integer `label` column, {tag: id}, idx_to_tag)."""
    mapping = {tag: i for i, tag in enumerate(LABELS)}
    df = df.copy()
    df[LABEL_ID_COLUMN] = df[COL_LABEL].map(mapping)
    if df[LABEL_ID_COLUMN].isna().any():
        raise ValueError(f"Labels outside the taxonomy: {sorted(set(df.loc[df[LABEL_ID_COLUMN].isna(), COL_LABEL]))}")
    df[LABEL_ID_COLUMN] = df[LABEL_ID_COLUMN].astype(int)
    present = sorted(set(df[COL_LABEL]))
    missing = [t for t in LABELS if t not in present]
    print(f"Using fixed taxonomy with {len(LABELS)} classes; present in data: {len(present)}."
          + (f" MISSING from data: {missing}" if missing else ""))
    return df, mapping, list(LABELS)
