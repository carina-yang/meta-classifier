"""
data_splitting.py
-----------------
Group-aware train/test/eval split (default: 70/15/15).

Why group-aware:
Column names repeat heavily across tables (e.g., 'email' appears in 40 tables,
'cust_email' normalizes to same group as 'email'). Random row splitting would
place the same column names in both train and test, inflating accuracy.

Solution:
Rows with matching normalized column names (lowercase, alphanumerics only)
form a GROUP. Each group assigned to exactly ONE split, ensuring test/eval
measure true generalization to unseen column names.

Algorithm:
Greedy assignment (largest groups first) to split with biggest class-weighted
deficit. Maintains per-class proportions close to target ratios. Classes with
insufficient samples are reported.
"""
import re
from typing import Tuple

import numpy as np
import pandas as pd

from config import COL_NAME, LABEL_ID_COLUMN


def group_key(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def split_data(
    df: pd.DataFrame, train_ratio: float, test_ratio: float, eval_ratio: float,
    min_samples_per_class: int, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + test_ratio + eval_ratio, 1.0):
        raise ValueError("train/test/eval ratios must sum to 1.0")
    df = df.copy()
    df["_group"] = df[COL_NAME].map(group_key)
    ratios = {"train": train_ratio, "test": test_ratio, "eval": eval_ratio}
    classes = sorted(df[LABEL_ID_COLUMN].unique())
    class_total = df[LABEL_ID_COLUMN].value_counts().to_dict()
    small = [c for c, n in class_total.items() if n < min_samples_per_class]
    if small:
        print(f"WARNING: label id(s) {small} have fewer than {min_samples_per_class} rows.")

    # group -> per-class row counts
    gc = df.groupby(["_group", LABEL_ID_COLUMN]).size().unstack(fill_value=0)
    rng = np.random.default_rng(seed)
    order = np.arange(len(gc))
    rng.shuffle(order)
    sizes = gc.sum(axis=1).to_numpy()
    order = order[np.argsort(-sizes[order], kind="stable")]  # big groups first, random ties

    current = {s: {c: 0 for c in classes} for s in ratios}
    assignment = {}
    counts = gc.to_numpy()
    cols = list(gc.columns)
    for gi in order:
        best, best_score = None, -np.inf
        for s, r in ratios.items():
            score = 0.0
            for ci, c in enumerate(cols):
                n = counts[gi, ci]
                if n:
                    target = r * class_total[c]
                    score += n * (target - current[s][c]) / (target + 1e-9)
            if score > best_score:
                best, best_score = s, score
        assignment[gc.index[gi]] = best
        for ci, c in enumerate(cols):
            current[best][c] += counts[gi, ci]

    split_of = df["_group"].map(assignment)
    out = {s: df[split_of == s].drop(columns="_group").reset_index(drop=True) for s in ratios}
    for s in ("test", "eval"):
        absent = [c for c in classes if (out[s][LABEL_ID_COLUMN] == c).sum() == 0]
        if absent:
            print(f"WARNING: label id(s) {absent} have NO rows in {s}.")
    absent = [c for c in classes if (out["train"][LABEL_ID_COLUMN] == c).sum() == 0]
    if absent:
        print(f"WARNING: label id(s) {absent} have NO rows in train.")
    print(f"Group-aware split sizes -> train: {len(out['train'])}, test: {len(out['test'])}, "
          f"eval: {len(out['eval'])} (of {len(df)} cleaned rows; {len(gc)} column-name groups).")
    return out["train"], out["test"], out["eval"]
