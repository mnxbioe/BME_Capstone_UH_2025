"""
Purpose
--------
converts raw TDT event markers (epocs) into a per-trial
DataFrame. Each row corresponds to a single stimulation event, with
timestamps, inter-stimulus intervals, burst grouping, and window
flags used for later feature extraction.
"""

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
    A container for named time windows relative to each epoc onset.
    These define the time intervals. later used for slicing LFP and stim data around
    each stimulus pulse.
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
    """
    For each stream record total length (# of samples)
    and sampling rate (Hz). to verify if a trial windows
    would fall outside the available data.
    """
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
    """True if [t0+pre, t0+post] is fully inside every required stream.
    prevents slicing errors when events occur too close to a blocks start/end
    """
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
    groups consecutive pulses into bursts
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
    Build a per-event table from a TDT block.

    Each row = one stimulus onset.
    Columns describe timing, burst grouping, and optional in-bounds flags.

    ------------------------------------------------------------
    Always included columns:
        trial       → integer trial index (0, 1, 2, ...)
        t0_sec      → onset time in seconds
        t1_sec      → offset time (if available, else NaN)
        isi_ms      → inter-stimulus interval (difference between successive t0s)
    
    Optional columns:
        burst_id        → group index for bursts (if burst_gap_ms set)
        in_bounds_<win> → boolean flags per window (if windows provided)
    
    ------------------------------------------------------------
    Args:
        tdt_obj : full TDT block object (from tdt.read_block)
        epoc_key : name of event store (default auto-selects using heuristics)
        windows : WindowSet defining the analysis windows to check
        require_streams_for_bounds : which streams must contain each window;
                                     defaults to [auto.lfp] + [auto.stim if present]
        burst_gap_ms : ISI threshold (ms) above which a new burst_id is created

    Returns:
        pandas.DataFrame  — one row per event with timing and validation metadata.
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
