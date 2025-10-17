# tdt_extract_and_plot.py
# One-stop extractor: auto-picks stores, computes trial features, saves CSV/NPZ, and plots.

import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tdt

# ========= USER PATHS =========
# ========= USER PATHS =========
# ========= USER PATHS =========
BLOCK_DIR = r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\data\raw\Exp_1\ExperimentBL-230918-022225"
OUT_DIR   = os.path.join(os.path.dirname(BLOCK_DIR), "_interim")


os.makedirs(OUT_DIR, exist_ok=True)

# ========= WINDOWS (sec) =========
BASELINE = (-0.200, 0.000)      # LFP baseline
RESPONSE = (0.0125, 0.100)      # evoked response window (12.5–100 ms)
STIM_WIN = (-0.001, 0.005)      # stim snippet around t0 (tight)

# ========= HELPERS =========
def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan

def pick_first(keys, preds):
    """Return first key where any predicate returns True."""
    for k in keys:
        for p in preds:
            if p(k):
                return k
    return None

def as_seconds_duration(d):
    """Handle timedelta or float."""
    return d.total_seconds() if hasattr(d, "total_seconds") else float(d)

def get_stream_data(stream):
    """Return fs (float) and 1-D data vector (averaged across channels if multich)."""
    fs = float(getattr(stream, "fs"))
    data = np.asarray(stream.data)
    if data.ndim == 2:
        # assume (nch, nsamp); average across channels for quicklook
        data = data.mean(axis=0)
    return fs, data

def slice_by_seconds(data, fs, t0, t1):
    """Slice 1-D data by time (sec) assuming index 0 == 0 sec."""
    i0 = max(0, int(np.floor(t0 * fs)))
    i1 = min(len(data), int(np.ceil(t1 * fs)))
    if i1 <= i0:
        return np.asarray([], dtype=float)
    return data[i0:i1]

# ========= LOAD BLOCK =========
print(f"\nReading TDT block:\n  {BLOCK_DIR}")
tdt_data = tdt.read_block(BLOCK_DIR)

duration_sec = as_seconds_duration(getattr(tdt_data.info, "duration", 0))
print(f"read from t=0s to t={duration_sec:.2f}s")
print("Streams:", list(tdt_data.streams.keys()))
print("Epocs  :", list(tdt_data.epocs.keys()))

# ========= AUTO-SELECT STORES =========
# Epoc: prefer PC*, then PT*, then U*
epoc_key = pick_first(
    tdt_data.epocs.keys(),
    preds=[
        lambda k: k.lower().startswith("pc"),
        lambda k: k.lower().startswith("pt"),
        lambda k: k.lower().startswith("u"),
        lambda k: True,
    ],
)
if epoc_key is None:
    raise RuntimeError("No epoc stores found.")

# Stim: prefer IZn1, then sSig, then sOut (exact match, case-insens)
lower_streams = {k.lower(): k for k in tdt_data.streams.keys()}
stim_key = None
for candidate in ("izn1", "ssig", "sout"):
    if candidate in lower_streams:
        stim_key = lower_streams[candidate]
        break
if stim_key is None:
    # optional fallback: any stream whose name looks like 'IZn*'
    stim_key = pick_first(tdt_data.streams.keys(), [lambda k: k.lower().startswith("izn")])
if stim_key is None:
    print("⚠️  No explicit stim stream found; continuing without stim features.")

# LFP: prefer LFP*, else Wav*
lfp_key = pick_first(
    tdt_data.streams.keys(),
    preds=[
        lambda k: "lfp" in k.lower(),
        lambda k: "wav" in k.lower(),
    ],
)
if lfp_key is None:
    raise RuntimeError("No LFP/Wav-like stream found (looked for 'LFP*' or 'Wav*').")

print(f"Auto-select → epoc={epoc_key}, lfp={lfp_key}, stim={stim_key}")

# ========= FETCH DATA ARRAYS =========
fs_lfp, lfp = get_stream_data(tdt_data.streams[lfp_key])
fs_stim, stim = (None, None)
if stim_key is not None:
    fs_stim, stim = get_stream_data(tdt_data.streams[stim_key])

onsets = np.asarray(tdt_data.epocs[epoc_key].onset, dtype=float)
n_events = onsets.size
print(f"fs_lfp={fs_lfp:.3f} Hz, events={n_events}")

# ========= TRIAL LOOP =========
rows, overlay_windows = [], []
for i, t0 in enumerate(onsets):
    # LFP windows
    bseg = slice_by_seconds(lfp, fs_lfp, t0 + BASELINE[0],  t0 + BASELINE[1])
    rseg = slice_by_seconds(lfp, fs_lfp, t0 + RESPONSE[0],  t0 + RESPONSE[1])

    # Skip if out of bounds
    if bseg.size == 0 or rseg.size == 0:
        continue

    # Stim features (optional)
    stim_mean = stim_abs_max = stim_rms = np.nan
    if stim is not None:
        sseg = slice_by_seconds(stim, fs_stim, t0 + STIM_WIN[0], t0 + STIM_WIN[1])
        if sseg.size:
            stim_mean    = float(np.mean(sseg))
            stim_abs_max = float(np.max(np.abs(sseg)))
            stim_rms     = rms(sseg)

    rows.append({
        "trial": i,
        "t0_sec": float(t0),
        "lfp_baseline_rms": rms(bseg),
        "lfp_response_rms": rms(rseg),
        "stim_mean": stim_mean,
        "stim_abs_max": stim_abs_max,
        "stim_rms": stim_rms,
    })

    # store short window for overlay plot (−50 ms → +150 ms)
    pre, post = -0.050, 0.150
    w = slice_by_seconds(lfp, fs_lfp, t0 + pre, t0 + post)
    if w.size:
        t = np.linspace(pre, post, w.size)
        overlay_windows.append((t, w))

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "trials_table.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Wrote {len(df)} trials to:\n  {csv_path}")

# ========= SAVE TENSORS (NPZ) =========
y = df[["lfp_response_rms"]].to_numpy(dtype=np.float32)
if "stim_abs_max" in df and not df["stim_abs_max"].isna().all():
    X_cols = ["stim_abs_max", "stim_rms", "lfp_baseline_rms"]
else:
    X_cols = ["lfp_baseline_rms"]
X = df[X_cols].to_numpy(dtype=np.float32)

# simple cfg bucket (ok if stim missing)
cfg = np.nan_to_num(np.round(df.get("stim_mean", pd.Series(np.zeros(len(df)))).to_numpy()*1000),
                    nan=-1).astype(np.int64)

npz_path = os.path.join(OUT_DIR, "dataset_tensors_v0.npz")
np.savez_compressed(npz_path, X=X, y=y, cfg=cfg, X_cols=np.array(X_cols, dtype=object))
print(f"✅ Wrote tensors: {npz_path}")

# ========= MANIFEST =========
manifest = {
    "block_dir": BLOCK_DIR,
    "epoc": epoc_key,
    "lfp_stream": lfp_key,
    "stim_stream": stim_key,
    "windows": {"baseline": BASELINE, "response": RESPONSE, "stim": STIM_WIN},
    "n_trials": int(len(df)),
    "fs": {"lfp": fs_lfp, "stim": fs_stim},
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("✅ Wrote manifest.json")

# ========= PLOTS =========
# 1) Full LFP trace with event lines
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(lfp))/fs_lfp, lfp, lw=0.5)
for t0 in onsets:
    plt.axvline(t0, color="r", alpha=0.15)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Full recording with event onsets")
plt.tight_layout()
plt.show()

# 2) Overlay epochs (−50 ms → +150 ms)
plt.figure(figsize=(6, 4))
for t, seg in overlay_windows:
    plt.plot(t*1000.0, seg, alpha=0.25)
plt.axvline(0, color='r', lw=1)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("LFP epochs (overlayed)")
plt.tight_layout()
plt.show()

# 3) Zoom first 0.5 s
plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(lfp))/fs_lfp, lfp, lw=0.7)
for t0 in onsets:
    if t0 > 0.5: break
    plt.axvline(t0, color="r", alpha=0.4)
plt.xlim(0, 0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Zoomed-in view (first 0.5 s)")
plt.tight_layout()
plt.show()

# 4) Correlations
if "stim_abs_max" in df and not df["stim_abs_max"].isna().all():
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(df["stim_abs_max"], df["lfp_response_rms"], alpha=0.5)
    plt.xlabel("Stim |abs|max")
    plt.ylabel("LFP response RMS")
    plt.title("Stim vs Response")
    plt.tight_layout()
    plt.show()
else:
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(df["lfp_baseline_rms"], df["lfp_response_rms"], alpha=0.5)
    plt.xlabel("Baseline RMS")
    plt.ylabel("Response RMS")
    plt.title("Baseline vs Response")
    plt.tight_layout()
    plt.show()

print("\n✅ Done. Files written to:")
print("  ", OUT_DIR)


import numpy as np
import matplotlib.pyplot as plt
fs_stim = 1017.25
t = np.arange(len(stim)) / fs_stim
plt.figure(figsize=(6,2))
plt.plot(t[:2000], stim[:2000])
plt.xlabel("Time (s)")
plt.ylabel("IZn1 amplitude")
plt.title("Raw stim monitor (first 2 s)")
plt.show()
print("min:", stim.min(), "max:", stim.max())

for k, s in tdt_data.streams.items():
    print(k, getattr(s, 'fs', None), getattr(s, 'scale', None),
          "min:", np.min(s.data), "max:", np.max(s.data))



          # --- Evoked LFP (clean) from Wav2: band-pass, blank artifact, baseline, mean±SEM ---

import numpy as np, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass(x, fs, lo=1, hi=300, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

# 1) Get high-rate neural channel
fs_hi = float(tdt_data.streams["Wav2"].fs)
x = np.asarray(tdt_data.streams["Wav2"].data)
if x.ndim > 1:  # if multichannel, quicklook = mean across chans
    x = x.mean(axis=0)

# 2) Light filtering
x_f = bandpass(x, fs_hi, lo=1, hi=300)

# 3) Epoch around each pulse
win = (-0.010, 0.080)  # -10 ms .. +80 ms
n0 = int(abs(win[0]) * fs_hi)
n1 = int(win[1] * fs_hi)
epochs = []
for t0 in onsets:
    i0 = int((t0 + win[0]) * fs_hi)
    i1 = i0 + n0 + n1
    if i0 >= 0 and i1 <= len(x_f):
        e = x_f[i0:i1].copy()

        # 4) Artifact blanking: zero out first 1.5 ms post-stim (t=0..1.5 ms)
        blank = int(0.0015 * fs_hi)
        e[n0:n0+blank] = e[n0-1] if n0 > 0 else 0.0

        # 5) Baseline subtract using -5..0 ms
        b0 = n0 - int(0.005 * fs_hi)
        base = e[b0:n0]
        e -= np.mean(base)

        epochs.append(e)

epochs = np.asarray(epochs)
t = np.linspace(win[0], win[1], epochs.shape[1]) * 1000  # ms

# 6) Mean ± SEM
mean = epochs.mean(axis=0)
sem  = epochs.std(axis=0) / np.sqrt(len(epochs))
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=str, default=BLOCK_DIR,
                    help="Path to TDT block folder (contains .tsq/.tev)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output folder (default: <block>/../_interim)")
    args = ap.parse_args()

    # override paths if provided
    if args.block:
        BLOCK_DIR = args.block
    if args.out:
        OUT_DIR = args.out
        os.makedirs(OUT_DIR, exist_ok=True)

    # (no other changes needed; the rest of the code uses BLOCK_DIR/OUT_DIR)

# 7) Plot (clean, non-flat evoked LFP)
plt.figure(figsize=(6,4))
# (optional) show a few single trials faintly
for k in np.linspace(0, len(epochs)-1, 20, dtype=int):
    plt.plot(t, epochs[k], alpha=0.15)

plt.plot(t, mean, lw=2)
plt.fill_between(t, mean-sem, mean+sem, alpha=0.25, linewidth=0)
plt.axvline(0, color='r', lw=1)
plt.xlim(-10, 80)
plt.ylim(-150, 150)  # adjust if needed
plt.xlabel("Time (ms)")
plt.ylabel("LFP (a.u. or µV if scale applied)")
plt.title("Evoked LFP (Wav2, band-passed, artifact-blanked)")
plt.tight_layout()
plt.show()


