from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Iterable, List
import numpy as np
import pandas as pd

from .tdt_reader import (
    auto_select_stores,
    get_event_onsets,
)

@dataclass(frozen=True)
class WindowSet:
    """
    A named collection of windows in seconds relative to epoc onset.
    Example:
        WindowSet({
            "overlay":  (-0.050, 0.150),
            "baseline": (-0.200, 0.000),
            "response": (0.0125, 0.100),
            "stim":     (-0.001, 0.005),
        })
    """
    windows: Mapping[str, Tuple[float, float]]

def _stream_lengths_and_fs(tdt_obj, streams: Iterable[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    lengths, fs_map = {}, {}
    for k in streams:
        st = tdt_obj.streams[k]
        data = np.asarray(st.data)
        n = data.shape[-1]  # (nch, n) or (n,)
        lengths[k] = int(n)
        fs_map[k] = float(getattr(st, "fs"))
    return lengths, fs_map

def _in_bounds_mask(
    onsets: np.ndarray,
    win: Tuple[float, float],
    lengths: Mapping[str, int],
    fs_map: Mapping[str, float],
    require_streams: Iterable[str],
) -> np.ndarray:
    """True if [t0+pre, t0+post] is fully inside every required stream."""
    pre, post = win
    ok = np.ones(onsets.size, dtype=bool)
    for s in require_streams:
        fs = fs_map[s]
        n  = lengths[s]
        i0 = np.floor((onsets + pre)  * fs).astype(int)
        i1 = np.ceil( (onsets + post) * fs).astype(int)
        ok &= (i0 >= 0) & (i1 <= n)
    return ok

def _burst_ids(isi_ms: np.ndarray, gap_ms: float) -> np.ndarray:
    """
    Assign a burst id that increments whenever the inter-stimulus interval exceeds gap_ms.
    First pulse gets burst_id 0.
    """
    burst = np.zeros_like(isi_ms, dtype=int)
    b = 0
    for i in range(1, isi_ms.size):
        if not np.isfinite(isi_ms[i]) or isi_ms[i] > gap_ms:
            b += 1
        burst[i] = b
    return burst

def make_trial_table(
    tdt_obj,
    epoc_key: Optional[str] = None,
    windows: Optional[WindowSet] = None,
    require_streams_for_bounds: Optional[List[str]] = None,
    burst_gap_ms: float = 5.0,
) -> pd.DataFrame:
    """
    Build a tidy per-event table from a TDT block.

    Columns (always):
      trial, t0_sec, t1_sec, isi_ms

    Optional columns:
      burst_id (if burst_gap_ms is not None)
      in_bounds_<winname> (if windows and require_streams_for_bounds provided)

    Args:
      epoc_key: which epoc store to use (default: auto-selects)
      windows:  named windows to check for in-bounds
      require_streams_for_bounds: list of streams that must all contain each window
                                  (default: [auto.lfp] + [auto.stim if present])
      burst_gap_ms: threshold to split bursts on ISI (None to disable)
    """
    picks = auto_select_stores(tdt_obj)
    epoc = epoc_key or picks.epoc

    onsets = np.asarray(get_event_onsets(tdt_obj, epoc), float)
    offsets = np.asarray(getattr(tdt_obj.epocs[epoc], "offset", []), float)
    if offsets.size != onsets.size:
        offsets = np.full_like(onsets, np.nan)

    # Inter-stimulus interval (ms)
    isi_ms = np.r_[np.nan, np.diff(onsets) * 1000.0]

    df = pd.DataFrame({
        "trial": np.arange(onsets.size, dtype=int),
        "t0_sec": onsets,
        "t1_sec": offsets,
        "isi_ms": isi_ms,
    })

    # Optional: burst grouping
    if burst_gap_ms is not None:
        df["burst_id"] = _burst_ids(isi_ms, gap_ms=float(burst_gap_ms))

    # In-bounds flags for requested windows
    if windows is not None:
        # By default, require LFP windows to be present in LFP (and stim in stim if present).
        if require_streams_for_bounds is None:
            req = [picks.lfp]
            if picks.stim:
                req.append(picks.stim)
            require_streams_for_bounds = req

        lengths, fs_map = _stream_lengths_and_fs(tdt_obj, require_streams_for_bounds)
        for wname, w in windows.windows.items():
            df[f"in_bounds_{wname}"] = _in_bounds_mask(
                onsets, w, lengths, fs_map, require_streams_for_bounds
            )

    return df
