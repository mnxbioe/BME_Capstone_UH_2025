# src/bme_capstone/dataio/target_builder.py
"""
Build training targets (y) and optional condition vector (cfg) from features.

Defaults:
- y := LFP response RMS  (shape [N,1], float32)
- cfg := rounded stim_mean * 1000 (int64), with -1 for NaN (stim OFF)
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

def build_tensors_v0(
    features: pd.DataFrame,
    input_cols: List[str] = None,
    target_col: str = "lfp_response_rms",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X  [N, d_in] float32
      y  [N, 1]    float32
      cfg [N]      int64
      X_cols list[str]
    """
    if input_cols is None:
        # If stim features exist, use them + baseline; else only baseline
        if "stim_abs_max" in features.columns and not features["stim_abs_max"].isna().all():
            input_cols = ["stim_abs_max", "stim_rms", "lfp_baseline_rms"]
        else:
            input_cols = ["lfp_baseline_rms"]

    # Drop rows with NaN in required columns
    req = list(set(input_cols + [target_col, "stim_mean"]))
    df = features[req].dropna().copy()

    X = df[input_cols].to_numpy(dtype=np.float32)
    y = df[[target_col]].to_numpy(dtype=np.float32)

    stim_mean = df.get("stim_mean", pd.Series(np.zeros(len(df))))
    cfg = np.nan_to_num(np.round(stim_mean.to_numpy()*1000), nan=-1).astype(np.int64)

    return X, y, cfg, input_cols
