# scripts/test_TDTreader.py
# -------------------------------------------------------------------------
# Purpose:
#   Quick standalone script to test the TDT reader pipeline.
#   It loads a TDT block, prints a summary, and optionally plots
#   peri-stimulus LFP overlays for visual inspection.
#
# Usage examples:
#   python scripts/test_reader.py
#   python scripts/test_reader.py --block "C:\path\to\ExperimentBL-XXXX" --window -0.1 0.2
#   python scripts/test_reader.py --save
# -------------------------------------------------------------------------

import sys, argparse
from pathlib import Path


def main():
    # ---------------------------------------------------------------
    # 1️⃣  Parse command-line arguments
    # ---------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Quick test for TDT reader")

    # Path to the TDT block (folder containing .tsq/.tev files)
    ap.add_argument(
        "--block", type=str, required=False,
        default=r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\data\raw\Exp_1\ExperimentBL-230918-001837",
        help="Path to TDT ExperimentBL-* folder"
    )

    # Time window (seconds before/after each event) for overlay plotting
    ap.add_argument(
        "--window", type=float, nargs=2, default=[-0.05, 0.15],
        metavar=("PRE", "POST"),
        help="Overlay window in seconds"
    )

    # Option to save plot to disk instead of showing it interactively
    ap.add_argument(
        "--save", action="store_true",
        help="Save plot to scripts/plots instead of show()"
    )

    args = ap.parse_args()
    BLOCK = args.block  # resolved block path from CLI or default

    # ---------------------------------------------------------------
    # 2️⃣  Add /src to Python path so imports work no matter where run
    # ---------------------------------------------------------------
    ROOT = Path(__file__).resolve().parents[1]   # repo root (one level up)
    SRC = ROOT / "src"                           # src/ folder path
    sys.path.append(str(SRC))
    print(f"[test_reader] Added to sys.path: {SRC}")
    
    # ---------------------------------------------------------------
    # 3️⃣  Import reader utilities from your package
    # ---------------------------------------------------------------
    from bme_capstone.dataio.tdt_reader import (
    read_block,
    quick_summary,
    auto_select_stores,   # ← needed for auto = auto_select_stores(blk)
    get_event_onsets,
    epoch_lfp,
    get_stream             # ← added for your plotting snippet
)
    # NEW: event indexer
    from bme_capstone.dataio.event_indexer import WindowSet, make_trial_table
    


    # ---------------------------------------------------------------
    # 4️⃣  Validate the input block directory before loading
    # ---------------------------------------------------------------
    blk_dir = Path(BLOCK)
    assert blk_dir.is_dir(), f"Not a folder: {blk_dir}"
    assert any(p.suffix.lower() == ".tsq" for p in blk_dir.iterdir()), "No .tsq in folder"
    assert any(p.suffix.lower() == ".tev" for p in blk_dir.iterdir()), "No .tev in folder"

    # ---------------------------------------------------------------
    # 5️⃣  Load block and print summary
    # ---------------------------------------------------------------
    print(f"[test_reader] Reading block from: {BLOCK}")
    blk = read_block(BLOCK)                 # core reader call
    summary = quick_summary(blk)            # basic stats / metadata
    print("[test_reader] Summary:")
    print(summary)

    # quick hint about whether stim is probably ON or OFF
    stim_hint = summary.get("stim_absmax_median")
    if stim_hint is not None:
        state = "OFF" if stim_hint == 0.0 else "ON?"
        print(f"[test_reader] Stim median |abs|max ≈ {stim_hint:.3g} → {state}")
        print("[test_reader] Summary:")
    print(summary)
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 5.5️⃣  Quick 4-panel sanity plot (overview + zoom + pulse stats)
    # ---------------------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    auto = auto_select_stores(blk)

    # --- Get raw LFP and stim streams ---
    fs_lfp, lfp_data, _ = get_stream(blk, auto.lfp)
    fs_stim, stim_data, _ = get_stream(blk, auto.stim)

    # --- Build time axes ---
    t_lfp = np.arange(lfp_data.shape[1]) / fs_lfp if lfp_data.ndim == 2 else np.arange(len(lfp_data)) / fs_lfp
    t_stim = np.arange(stim_data.shape[1]) / fs_stim if stim_data.ndim == 2 else np.arange(len(stim_data)) / fs_stim

    # --- Index for 2-second and 10-ms views ---
    dur_overview = 2.0
    dur_zoom = 0.01
    lfp_idx_full = t_lfp <= dur_overview
    stim_idx_full = t_stim <= dur_overview
    lfp_idx_zoom = t_lfp <= dur_zoom
    stim_idx_zoom = t_stim <= dur_zoom

    # --- Pick first channel if multichannel ---
    lfp_trace = lfp_data[0] if lfp_data.ndim == 2 else lfp_data
    stim_trace = stim_data[0] if stim_data.ndim == 2 else stim_data

    # --- Compute simple pulse stats for stim ---
    stim_abs = np.abs(stim_trace)
    peak_amp = np.max(stim_abs[stim_idx_zoom])
    # count zero-crossings to estimate pulse frequency
    zcross = np.where(np.diff(np.sign(stim_trace[stim_idx_zoom])) != 0)[0]
    n_pulses = len(zcross) // 2
    pulse_rate = n_pulses / dur_zoom  # Hz (approx.)
    print(f"[stim-check] peak ≈ {peak_amp:.1f}  |  pulses ≈ {n_pulses}  |  est. freq ≈ {pulse_rate:.1f} Hz")

    # --- Create 4-panel figure ---
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=False)
    # Choose correct factor for your system (adjust if needed)
    SCALE_NA_PER_UNIT = 3.05      # ≈ nA per raw unit (IZ2-64 default)
    stim_phys = stim_trace * SCALE_NA_PER_UNIT * 1e-3  # convert to µA


    # (1) LFP 2 s
    axes[0].plot(t_lfp[lfp_idx_full], lfp_trace[lfp_idx_full])
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{auto.lfp} (neural, 0–2 s)")

    # (2) Stim 2 s
    axes[1].plot(t_stim[stim_idx_full], stim_phys[stim_idx_full])
    axes[1].set_ylabel("Current (µA)")

    axes[1].set_title(f"{auto.stim} (stimulation, 0–2 s)")

    # (3) LFP zoom 10 ms
    axes[2].plot(t_lfp[lfp_idx_zoom]*1e3, lfp_trace[lfp_idx_zoom])
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title(f"{auto.lfp} (zoom, first {dur_zoom*1e3:.0f} ms)")

    # (4) Stim zoom 10 ms + annotate pulses
    axes[3].plot(t_stim[stim_idx_zoom]*1e3, stim_phys[stim_idx_zoom])
    axes[3].set_ylabel("Current (µA)")
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_title(f"{auto.stim} (zoom, first {dur_zoom*1e3:.0f} ms)")

    axes[3].text(
        0.01, 0.9, f"peak={peak_amp:.0f},  pulses={n_pulses},  ~{pulse_rate:.0f} Hz",
        transform=axes[3].transAxes, fontsize=9, color="darkgreen"
    )

    fig.tight_layout()

    # --- Save or show ---
    if args.save:
        outdir = Path(__file__).resolve().parent / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"sanity_{Path(BLOCK).name}.png"
        fig.savefig(outpath, dpi=150)
        print(f"[test_reader] Saved sanity plot → {outpath}")
    else:
        plt.show()

    # --- NEW: build a tidy trial table ---
    wins = WindowSet({
     "overlay":  (-0.050, 0.150),
     "baseline": (-0.200, 0.000),
     "response": (0.0125, 0.100),
     "stim":     (-0.001, 0.005),
    })
    tt = make_trial_table(blk, windows=wins, burst_gap_ms=5.0)
    print(f"[test_reader] Trial table: {tt.shape[0]} rows")
    print(tt.head(5))

    # optional: how many trials are fully in-bounds per window?
    inb_cols = [c for c in tt.columns if c.startswith("in_bounds_")]
    if inb_cols:
        frac = tt[inb_cols].mean().sort_index()
        print("[test_reader] In-bounds fractions:")
        for k, v in frac.items():
            print(f"  {k}: {v:.3f}")

    from bme_capstone.dataio.feature_bank import compute_features
    from bme_capstone.dataio.target_builder import build_tensors_v0

    wins = dict(
        overlay=(-0.050, 0.150),
        baseline=(-0.200, 0.000),
        response=(0.0125, 0.100),
        stim=(-0.001, 0.005),
    )

    # compute features on in-bounds trials
    feat = compute_features(blk, tt, windows=wins, require_inbounds=True)
    print(f"[features] rows={len(feat)}, cols={list(feat.columns)}")
    print(feat.head(3))

    # build tensors
    X, y, cfg, X_cols = build_tensors_v0(feat)
    print(f"[tensors] X={X.shape}, y={y.shape}, cfg={cfg.shape}, X_cols={X_cols}")
        
    # ---------------------------------------------------------------
    # 6️⃣  Extract aligned epochs for LFP overlays
    # ---------------------------------------------------------------
    auto = auto_select_stores(blk)                      # pick epoc/LFP/stim stores
    onsets = get_event_onsets(blk, auto.epoc)           # event onset times (s)
    pre, post = args.window                             # pre/post window limits
    t, E = epoch_lfp(blk, onsets, auto.lfp, (pre, post))# extract LFP segments
    print(f"[test_reader] Epochs: {E.shape} using window ({pre:+.3f}, {post:+.3f}) s")

    # ---------------------------------------------------------------
    # 7️⃣  Plot overlays (or save to file)
    # ---------------------------------------------------------------
    # 7️⃣  Plot overlays (baseline-subtracted, shaded windows, mean±SEM, robust limits)
    # ---------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
       

        if E.size:
            # --- analysis windows (seconds) ---
            BASE = (-0.050, 0.000)
            RESP = (0.0125, 0.100)

            # masks on the epoch time vector t (seconds)
            base_m = (t >= BASE[0]) & (t < BASE[1])
            resp_m = (t >= RESP[0]) & (t < RESP[1])

            # --- baseline-subtract each trial (-50..0 ms mean) ---
            E_bs = E - E[:, base_m].mean(axis=1, keepdims=True)

            # how many trials to overlay (spaghetti)
            k = min(100, E_bs.shape[0])
            t_ms = t * 1e3

            fig, ax = plt.subplots(figsize=(7.5, 4.2))

            # shade analysis windows for clarity
            ax.axvspan(BASE[0]*1e3, BASE[1]*1e3, alpha=0.12, label="Baseline window")
            ax.axvspan(RESP[0]*1e3, RESP[1]*1e3, alpha=0.12, label="Response window")

            # spaghetti (baseline-subtracted)
            ax.plot(t_ms, E_bs[:k].T, alpha=0.06, lw=0.7)

            # mean ± SEM
            mu  = E_bs.mean(axis=0)
            sem = E_bs.std(axis=0, ddof=1) / np.sqrt(E_bs.shape[0])
            ax.plot(t_ms, mu, color="black", lw=1.5, label="Mean")
            ax.fill_between(t_ms, mu-sem, mu+sem, color="gray", alpha=0.25, label="±1 SEM")

            # stim onset line
            ax.axvline(0, color="r", lw=1)

            # robust y-limits around [-10, +20] ms region
            focus = (t >= -0.010) & (t <= 0.020)
            lo, hi = np.percentile(E_bs[:, focus], [1, 99])
            pad = 0.1 * max(1e-12, hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

            # quick effect summary over windows
            base_trial = E_bs[:, base_m].mean(axis=1)
            resp_trial = E_bs[:, resp_m].mean(axis=1)
            delta = resp_trial  # baseline-subtracted → Δ = response mean
            d_mu  = float(delta.mean())
            d_sem = float(delta.std(ddof=1) / np.sqrt(delta.size))
            ax.text(0.98, 0.02, f"Δμ={d_mu:.3g} ± {d_sem:.3g}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="dimgray")

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("LFP (baseline-subtracted, raw units)")
            ax.set_title(f"{auto.lfp} overlays (n={k}/{E_bs.shape[0]})")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            ax.grid(alpha=0.15, linestyle=":")

            fig.tight_layout()
            if args.save:
                outdir = Path(__file__).resolve().parent / "plots"
                outdir.mkdir(parents=True, exist_ok=True)
                outpath = outdir / f"overlay_{Path(BLOCK).name}.png"
                fig.savefig(outpath, dpi=150)
                print(f"[test_reader] Saved plot → {outpath}")
            else:
                plt.show()
        else:
            print("[test_reader] No in-bounds epochs for window.")
    except Exception as e:
        print(f"[test_reader] Plot skipped: {e}")

    

# ---------------------------------------------------------------
# 8️⃣  Entry point guard (so main() runs only when executed directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
