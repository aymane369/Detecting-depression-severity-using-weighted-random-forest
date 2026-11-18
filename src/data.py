"""Data loading, target construction and data-regime definition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Read the raw CSV (2,621 rows, one row per visit)."""
    return pd.read_csv(C.DATA_PATH)


def shared_value_mask(df: pd.DataFrame, cols: list[str] = C.BIOMARKERS) -> pd.Series:
    """True for rows where at least one biomarker value also occurs in another row."""
    shared = pd.Series(False, index=df.index)
    for c in cols:
        shared |= df.groupby(c)[c].transform("size") > 1
    return shared


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encoding of the categorical columns (same convention as the article).

    yes/no -> 1/0 ; Gender F/M -> 0/1 ; ACE haplotype '1','2','3','3+' -> 1,2,3.
    """
    out = df.copy()
    for c in ["OtherMeds-Depression", "DM-Status", "DM-MedUse", "CVD-Revised", "HT-Status", "HT-MedUse"]:
        out[c] = (out[c].str.lower() == "yes").astype(int)
    out["Gender"] = (out["Gender"] == "M").astype(int)
    out["ACE-Update"] = out["ACE-Update"].astype(str).str.replace("+", "", regex=False).astype(int)
    return out


def load_dataset(regime: str = "full") -> pd.DataFrame:
    """Return the encoded dataset for a data regime.

    full  : all rows.
    clean : only rows whose Humanin, MOTSc and P66 values are all unique in the raw file
            (proxy for measured, non-duplicated values; see the leakage audit).
    """
    raw = load_raw()
    if regime == "full":
        df = raw
    elif regime == "clean":
        df = raw.loc[~shared_value_mask(raw)]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    df = encode(df).reset_index(drop=True)
    df["y_binary"] = (df[C.TARGET_COL] > 4).astype(int)
    df["y_multi"] = pd.cut(df[C.TARGET_COL], C.PHQ_BINS, labels=False).astype(int)
    return df


def get_xy(regime: str, task: str, fset: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_dataset(regime)
    y = df["y_binary" if task == "binary" else "y_multi"].to_numpy()
    return df[C.FEATURE_SETS[fset]].copy(), y


def holdout_split(X: pd.DataFrame, y: np.ndarray, regime: str, task: str, seed: int = C.SEED):
    """70/30 stratified split. The stratification variable is identical for both tasks
    of a given regime so that binary and multiclass experiments share the same split."""
    df = load_dataset(regime)
    strat = df["y_multi"] if regime == "full" else df["y_binary"]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=strat, random_state=seed)


def class_names(task: str) -> list[str]:
    return C.CLASS_NAMES_BINARY if task == "binary" else C.CLASS_NAMES_MULTI
