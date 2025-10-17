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
        read_block, quick_summary,
        auto_select_stores, get_event_onsets, epoch_lfp
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
    try:
        import matplotlib.pyplot as plt

        if E.size:
            k = min(100, E.shape[0])      # plot up to 100 trials for clarity
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.plot(t * 1e3, E[:k].T, alpha=0.08)   # convert s → ms
            ax.axvline(0, color="r", lw=1)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("LFP (raw units)")
            ax.set_title(f"{auto.lfp} overlays (n={k}/{E.shape[0]})")

            # Save or show depending on user flag
            if args.save:
                outdir = Path(__file__).resolve().parent / "plots"
                outdir.mkdir(parents=True, exist_ok=True)
                outpath = outdir / f"overlay_{Path(BLOCK).name}.png"
                fig.tight_layout()
                fig.savefig(outpath, dpi=150)
                print(f"[test_reader] Saved plot → {outpath}")
            else:
                fig.tight_layout()
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
