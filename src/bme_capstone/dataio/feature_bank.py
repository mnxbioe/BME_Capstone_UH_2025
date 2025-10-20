# src/bme_capstone/dataio/feature_bank.py
"""
Small feature set for v0 ( for testing dataio):
- LFP baseline RMS
- LFP response RMS
- (if available) stim window: mean, abs_max, RMS

Inputs:
  - tdt_obj: from tdt_reader.read_block()
  - trial_table: from event_indexer.make_trial_table()
  - windows: WindowSet (uses 'baseline', 'response', 'stim' if present)
  - lfp_name / stim_name: explicit stores (defaults: auto-select)

Returns:
  pandas.DataFrame with one row per trial and a 'trial' column.
"""

from __future__ import annotations
from typing import Optional, Tuple, Mapping, Dict
import numpy as np
import pandas as pd

from .tdt_reader import auto_select_stores, get_stream

def _mean_across_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x.mean(axis=0)
    return x

def _slice_by_seconds(data: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(np.floor(t0 * fs)))
    i1 = min(len(data), int(np.ceil(t1 * fs)))
    if i1 <= i0:
        return np.asarray([], dtype=float)
    return data[i0:i1]

def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan

def compute_features(
    tdt_obj,
    trial_table: pd.DataFrame,
    windows: Mapping[str, Tuple[float, float]],
    lfp_name: Optional[str] = None,
    stim_name: Optional[str] = None,
    require_inbounds: bool = True,
) -> pd.DataFrame:
    """
    Compute per-trial scalar features from LFP and (optional) stim.
    If require_inbounds=True, keeps only rows where baseline/response (and stim if present) are in-bounds.
    """
    picks = auto_select_stores(tdt_obj)
    lfp_key  = lfp_name  or picks.lfp
    stim_key = stim_name or picks.stim

    fs_lfp, lfp_data, _ = get_stream(tdt_obj, lfp_key)
    lfp_1d = _mean_across_channels(lfp_data)

    fs_stim, stim_1d = None, None
    if stim_key:
        fs_stim, stim_data, _ = get_stream(tdt_obj, stim_key)
        stim_1d = _mean_across_channels(stim_data)

    # optional row filtering using in-bounds flags (if present)
    tt = trial_table.copy()
    need_flags = []
    for name in ("baseline", "response"):
        if name in windows:
            need_flags.append(f"in_bounds_{name}")
    if stim_1d is not None and "stim" in windows:
        need_flags.append("in_bounds_stim")

    if require_inbounds and need_flags:
        have_all = [c for c in need_flags if c in tt.columns]
        if have_all:
            mask = np.logical_and.reduce([tt[c].values.astype(bool) for c in have_all])
            tt = tt.loc[mask].reset_index(drop=True)

    rows = []
    base_win = windows.get("baseline")
    resp_win = windows.get("response")
    stim_win = windows.get("stim")

    for i, r in tt.iterrows():
        t0 = float(r["t0_sec"])

        # LFP features
        base_rms = resp_rms = np.nan
        if base_win is not None:
            b = _slice_by_seconds(lfp_1d, fs_lfp, t0 + base_win[0], t0 + base_win[1])
            base_rms = _rms(b)
        if resp_win is not None:
            e = _slice_by_seconds(lfp_1d, fs_lfp, t0 + resp_win[0], t0 + resp_win[1])
            resp_rms = _rms(e)

        # Stim features (optional)
        stim_mean = stim_abs_max = stim_rms = np.nan
        if stim_1d is not None and stim_win is not None:
            s = _slice_by_seconds(stim_1d, fs_stim, t0 + stim_win[0], t0 + stim_win[1])
            if s.size:
                stim_mean    = float(np.mean(s))
                stim_abs_max = float(np.max(np.abs(s)))
                stim_rms     = _rms(s)

        rows.append(dict(
            trial=int(r["trial"]),
            t0_sec=t0,
            lfp_baseline_rms=base_rms,
            lfp_response_rms=resp_rms,
            stim_mean=stim_mean,
            stim_abs_max=stim_abs_max,
            stim_rms=stim_rms,
        ))

    return pd.DataFrame(rows)

# === Tower A helper: per-electrode current vectors ============================
from typing import List

def compute_stim_currents_table(
    tdt_obj,
    trial_table: pd.DataFrame,
    stim_name: Optional[str] = None,
    first_phase_sec: float = 0.0005,  # 0.5 ms; covers typical first phase (~200 µs)
) -> pd.DataFrame:
    """
    Build I^{(n)} for Tower A: one signed scalar per electrode, per trial.
    We estimate the signed amplitude from the *leading phase* of each pulse:
      1) slice [t0, t0 + first_phase_sec]
      2) find index of max |value|
      3) use that sample's sign * magnitude as I_j^{(n)}

    This gives a physically meaningful current per electrode suitable for
    Neumann BCs (source/sink) in the Laplace PINN.

    Returns:
      DataFrame with columns:
        trial, t0_sec, I_ch00, I_ch01, ..., I_ch{nch-1}
      If no stim stream exists, returns an empty DataFrame with just trial/t0_sec.
      Charge-balanced biphasic pulses have near-zero mean, so we can’t use mean.
    """
    picks = auto_select_stores(tdt_obj)
    stim_key = stim_name or picks.stim
    if not stim_key:
        # No stim in this block → return trial/t0 only
        return trial_table[["trial", "t0_sec"]].copy()

    fs_stim, stim_data, _ = get_stream(tdt_obj, stim_key)  # shape: (nch, n) or (n,)
    stim_arr = np.asarray(stim_data)

    # Ensure 2-D: (nch, n_samples)
    if stim_arr.ndim == 1:
        stim_arr = stim_arr[None, :]

    nch, _ = stim_arr.shape
    # Precompute column names to keep geometry ordering stable
    I_cols: List[str] = [f"I_ch{idx:02d}" for idx in range(nch)]

    rows = []
    for _, r in trial_table.iterrows():
        t0 = float(r["t0_sec"])
        # Slice only the *leading phase* window
        i0 = max(0, int(np.floor(t0 * fs_stim)))
        i1 = max(i0, int(np.ceil((t0 + first_phase_sec) * fs_stim)))

        I_vec = np.full(nch, np.nan, dtype=float)
        for ch in range(nch):
            s = stim_arr[ch, i0:i1]
            if s.size == 0:
                continue
            # signed amplitude = sign(sample_at_absmax) * absmax
            k = int(np.argmax(np.abs(s)))
            I_vec[ch] = float(np.sign(s[k]) * np.abs(s[k]))

        row = {"trial": int(r["trial"]), "t0_sec": t0}
        row.update({col: I_vec[i] for i, col in enumerate(I_cols)})
        rows.append(row)

    return pd.DataFrame(rows, columns=["trial", "t0_sec"] + I_cols)
