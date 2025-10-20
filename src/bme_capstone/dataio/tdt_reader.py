# src/bme_capstone/dataio/tdt_reader.py
"""
Purpose:
──────────────────────────────────────────────
TDT blocks store all signals recorded during an experiment:
  • LFP / Wav streams  -> neural signals
  • Stim streams       -> stimulation currents (IZn1 / sSig / sOut)
  • Epoc stores        -> event markers (PC1_ / PT1_)

This file standardizes how those signals are accessed and named
──────────────────────────────────────────────
Main features
──────────────────────────────────────────────
1. **read_block(path)**  
   Loads a full TDT block from disk

2. **auto_select_stores(tdt_obj)**  
   Automatically identifies the most likely epoc, LFP, and stim stores
   using name heuristics 

3. **get_stream(tdt_obj, name)**  
   Returns the sampling rate, waveform array, and scale factorr

4. **get_event_onsets(tdt_obj, epoc_name)**  
   Extracts onset times for stimulation or task events.

5. **quick_summary(tdt_obj)**  
   Generates a dictionary of block-level metadata
   (duration, store counts, event count, stim amplitude statistics...)
   — used for sanity checks, logs, and data indexing.

6. **epoch_lfp(...)**  
   extracts LFP segments around each event for visualization.

All params are defined inside this file for now.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable
import numpy as np

# external dependency: tdt (TDTpy)
import tdt


# ---------- defaults (local-only for now) ----------
# epoch windows used by higher-level feature extraction / quick sanity checks
BASELINE = (-0.200, 0.000)      # seconds window (Used to compute RMS baseline LFP before stimulation.)
RESPONSE = (0.0125, 0.100)      # Matches Dr Francis paper, used in RMS calculation
STIM_WIN = (-0.001, 0.005)      # extracted snippet length = 6 ms total.


# ---------- small utilities ( helper functions) ----------
#Convert any duration-like object to a float in seconds to ensures downstream code receives a numeric duration.: 
def _as_seconds_duration(d) -> float:
    return d.total_seconds() if hasattr(d, "total_seconds") else float(d)

#generic pattern matcher (used in auto_select_store)
def _first_match(keys: Iterable[str], preds: Iterable) -> Optional[str]:
    for k in keys:                         #A list of all store names in a TDT block
        for p in preds:                    #lsit of true of false predicated outputs 
            if p(k):
                return k                   # moment one predicate returns True return that key
    return None                            # picks the most appropriate epoc (PC1/ → PT1/ → U11/) 

#collapse multi-channel data into a single avg (used for computing median stim across channels)
def _mean_across_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x.mean(axis=0)  #Computes column wise mean
    return x

#extract a segment based on time(s),instead of sample indices.
def _slice_by_seconds(data: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    # Convert the time to a sample index.:
     # Multiply by sampling rate (fs) → seconds → samples.
     # floor() to include starting sample.
     # max() ensures the index is not negative.
    i0 = max(0, int(np.floor(t0 * fs)))
    # ceil() to ensure we include the last sample within the window.
    # min(len(data), ...) ensures the index does not exceed array length.
    i1 = min(len(data), int(np.ceil(t1 * fs)))
    if i1 <= i0:
        return np.asarray([], dtype=float) # empty array if wind is invalid
    return data[i0:i1]

#Compute RMS 
def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan


# ---------- public datatypes ----------
#Holds the automatically detected store names from a TDT block.
@dataclass
class AutoStores:
    epoc: str
    lfp: str
    stim: Optional[str]  # may be None if stim stream absent


@dataclass
class StreamInfo:
    fs: float
    shape: Tuple[int, ...]
    scale: Optional[float]

@dataclass
class BlockSummary:
    path: str
    duration_sec: float
    n_streams: int
    n_epocs: int
    auto: AutoStores
    n_events: int
    stim_absmax_median: float
    streams: Dict[str, StreamInfo]
    epocs: Dict[str, Dict[str, int]]


# ---------- core API ----------
def read_block(path: str):
    td = tdt.read_block(path)
    # Make sure we have a block path for summaries
    try:
        _ = getattr(td.info, "blockpath")
    except Exception:
        try:
            setattr(td.info, "blockpath", str(path))
        except Exception:
            pass
    return td


#Takes argument: tdt_obj, which is the block object returned by tdt.read_block(path).
#will return an AutoStores dataclass
def auto_select_stores(tdt_obj) -> AutoStores:
    """
    Heuristics(RULES):
      epoc: prefer PC*, then PT*, then U*
      lfp : prefer 'LFP*', else 'Wav*'
      stim: prefer exact 'IZn1', else 'sSig', else 'sOut', else first 'IZn*'
    """
    epoc_key = _first_match(
        tdt_obj.epocs.keys(),
        preds=[ #rules
            lambda k: k.lower().startswith("pc"),
            lambda k: k.lower().startswith("pt"),
            lambda k: k.lower().startswith("u"),
            lambda k: True,  # fallback: any epoc
        ],
    )
    if epoc_key is None:
        raise RuntimeError("No epoc stores found in block.")

    lfp_key = _first_match(
        tdt_obj.streams.keys(),
        preds=[lambda k: k.lower() == "wav1", lambda k: "lfp" in k.lower(), lambda k: k.lower().startswith("wav")],#rules
    )
    if lfp_key is None:
        raise RuntimeError("No LFP/Wav-like stream found.")

    lower_streams = {k.lower(): k for k in tdt_obj.streams.keys()}
    stim_key = None
    for cand in ("izn1", "ssig", "sout"):
        if cand in lower_streams:
            stim_key = lower_streams[cand]
            break
    if stim_key is None:
        stim_key = _first_match(tdt_obj.streams.keys(), [lambda k: k.lower().startswith("izn")])

    return AutoStores(epoc=epoc_key, lfp=lfp_key, stim=stim_key)


def get_stream(tdt_obj, name: str) -> Tuple[float, np.ndarray, Optional[float]]:
    """
    Return (fs, data, scale) for a given stream store.
    Data is returned as a numpy array with original shape from TDTpy.
    """
    s = tdt_obj.streams[name]
    fs = float(getattr(s, "fs"))
    data = np.asarray(s.data)
    scale = getattr(s, "scale", None)
    return fs, data, scale


def get_event_onsets(tdt_obj, epoc_name: str) -> np.ndarray:
    """Return 1-D numpy array of epoc onset times (seconds)."""
    return np.asarray(tdt_obj.epocs[epoc_name].onset, dtype=float)


def quick_summary(tdt_obj) -> dict:
    """
    Produce a compact dict for test and logs.
    """
    duration = _as_seconds_duration(getattr(tdt_obj.info, "duration", 0.0))
    auto = auto_select_stores(tdt_obj)

    # stream metadata
    streams = {}
    for k, s in tdt_obj.streams.items():
        streams[k] = dict(
            fs=float(getattr(s, "fs", np.nan)),
            shape=list(np.asarray(s.data).shape),
            scale=getattr(s, "scale", None),
        )

    # epoc counts
    epocs = {k: {"n": int(len(v.onset))} for k, v in tdt_obj.epocs.items()}

    # events + stim quick metric
    onsets = get_event_onsets(tdt_obj, auto.epoc) if auto.epoc else np.array([])
    n_events = int(onsets.size)

    stim_absmax_median = 0.0
    if auto.stim is not None:
        fs_stim, stim, _ = get_stream(tdt_obj, auto.stim)
        vals = []
        for t0 in onsets:
            seg = _slice_by_seconds(_mean_across_channels(stim), fs_stim,
                                    t0 + STIM_WIN[0], t0 + STIM_WIN[1])
            if seg.size:
                vals.append(float(np.max(np.abs(seg))))
        stim_absmax_median = float(np.median(vals)) if vals else 0.0

    return dict(
        path=str(getattr(tdt_obj.info, "blockpath", "")) or "UNKNOWN",
        duration_sec=float(duration),
        n_streams=len(streams),
        n_epocs=len(epocs),
        auto=dict(epoc=auto.epoc.replace("/", "_"),
                  lfp=auto.lfp,
                  stim=auto.stim),
        n_events=n_events,
        stim_absmax_median=stim_absmax_median,
        streams=streams,
        epocs=epocs,
    )



# ---------- convenience: epoch LFP (for quicklooks) ----------
def epoch_lfp(tdt_obj, onsets: np.ndarray, lfp_name: str, window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single-channel LFP epochs by averaging across channels
    and returning (time_vector_sec, epochs [n_trials, n_samples]).
    This is only for quick sanity plots; real feature extraction happens later.
    """
    fs, data, _ = get_stream(tdt_obj, lfp_name)
    x = _mean_across_channels(data)
    pre, post = window
    n_samples = int(round((post - pre) * fs))
    t = np.linspace(pre, post, n_samples, endpoint=False)
    ep = []
    for t0 in onsets:
        i0 = int(round((t0 + pre) * fs))
        i1 = i0 + n_samples
        if i0 < 0 or i1 > len(x):
            continue
        e = x[i0:i1]
        if len(e) == n_samples:
            ep.append(e)
    return t, np.asarray(ep, dtype=float)
