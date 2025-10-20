# scripts/build_dataset.py
# Batch-convert all ExperimentBL-* blocks under a root into per-block outputs.
# Outputs go to: data/interim/<session>/<block>/...
# Optional aggregate across blocks goes to: data/processed/<session>/...

from __future__ import annotations
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from inspect import signature

# --- Make src importable no matter where we run this script ---
ROOT = Path(__file__).resolve().parents[1]   # repo root
SRC  = ROOT / "src"

os.chdir(ROOT)                               # ensure working dir = repo root
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SRC / "bme_capstone") not in sys.path:
    sys.path.insert(0, str(SRC / "bme_capstone"))


# --- our package bits ---
from bme_capstone.dataio.tdt_reader import (
    read_block, quick_summary, auto_select_stores, epoch_lfp
)
from bme_capstone.dataio.event_indexer import (
    WindowSet, make_trial_table
)

from bme_capstone.dataio.feature_bank import (
    compute_features,
    compute_stim_currents_table,   # ← add this
)

from bme_capstone.dataio.target_builder import build_tensors_v0

# optional (stim-fallback) – present if you added it
try:
    from bme_capstone.dataio.event_indexer import get_onsets_with_fallback
    HAVE_FALLBACK = True
except Exception:
    HAVE_FALLBACK = False


def find_blocks(root: Path, pattern: str = "ExperimentBL-*") -> list[Path]:
    """Return all TDT block folders that contain .tsq/.tev."""
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    blocks = []
    for p in sorted(root.glob(pattern)):
        if p.is_dir() and any(x.suffix.lower()==".tsq" for x in p.iterdir()) and any(x.suffix.lower()==".tev" for x in p.iterdir()):
            blocks.append(p)
    return blocks


def process_block(block_dir: Path,
                  out_dir: Path,
                  save_plot: bool = False,
                  min_epocs_for_fallback: int = 10) -> dict:
    """
    Run reader → indexer → features → tensors for one block and write outputs.
    Returns a dict of small stats for aggregation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) read + summary
    blk = read_block(str(block_dir))
    smry = quick_summary(blk)

    # 2) windows
    wins = WindowSet({
        "overlay":  (-0.050, 0.150),
        "baseline": (-0.200, 0.000),
        "response": (0.0125, 0.100),
        "stim":     (-0.001, 0.005),
    })

    # 3) trial table
    used_fallback = False
    tt = None

    # make_trial_table may or may not accept 'onsets'/'use_fallback_if_sparse'
    sig = signature(make_trial_table)

    if HAVE_FALLBACK:
        # get onsets with fallback and pass explicitly if supported
        auto = auto_select_stores(blk)
        on = get_onsets_with_fallback(blk, epoc_key=auto.epoc, min_epocs=min_epocs_for_fallback)
        used_fallback = (len(on) > 0 and smry["epocs"].get(smry["auto"]["epoc"], {"n": 0})["n"] < min_epocs_for_fallback)
        if "onsets" in sig.parameters:
            tt = make_trial_table(blk, windows=wins, onsets=on, use_fallback_if_sparse=False)
        else:
            tt = make_trial_table(blk, windows=wins)
    else:
        # no fallback available, just epocs
        tt = make_trial_table(blk, windows=wins)

    # write trials
    trials_csv = out_dir / "trials.csv"
    tt.to_csv(trials_csv, index=False)

    # 3.5) Tower A currents table (per-electrode I^{(n)})
    auto = auto_select_stores(blk)
    I_df = compute_stim_currents_table(blk, tt, stim_name=auto.stim, first_phase_sec=0.0005)
    I_csv = out_dir / "currents_table.csv"
    I_df.to_csv(I_csv, index=False)

    # Also save as NPZ for easy loading in Torch/PINA
    I_cols = [c for c in I_df.columns if c.startswith("I_ch")]
    I_np  = I_df[I_cols].to_numpy(dtype=np.float32) if I_cols else np.zeros((len(I_df), 0), np.float32)
    np.savez_compressed(out_dir / "currents_table.npz",
                        trial=I_df["trial"].to_numpy(np.int64),
                        t0_sec=I_df["t0_sec"].to_numpy(np.float64),
                        I=I_np,
                        I_cols=np.array(I_cols, dtype=object))


    # 4) features (in-bounds filtering baked in)
    # unwrap WindowSet into a plain mapping for compute_features
    wmap = wins.windows if hasattr(wins, "windows") else wins
    feat = compute_features(blk, tt, windows=wmap, require_inbounds=True)

    feat_csv = out_dir / "features.csv"
    feat.to_csv(feat_csv, index=False)

    # 5) tensors (choose columns; drop NaNs)
    #    If stim is effectively off, fall back to baseline-only inputs.
    X, y, cfg, X_cols = build_tensors_v0(feat)
    if ("stim_abs_max" in feat.columns
        and np.allclose(feat["stim_abs_max"].fillna(0).to_numpy(), 0)
        and np.allclose(feat["stim_rms"].fillna(0).to_numpy(), 0)):
        X, y, cfg, X_cols = build_tensors_v0(feat, input_cols=["lfp_baseline_rms"])

    npz_path = out_dir / "dataset_tensors_v0.npz"
    np.savez_compressed(npz_path, X=X, y=y, cfg=cfg, X_cols=np.array(X_cols, dtype=object))

    # 6) manifest
    man = {
        "block_dir": str(block_dir),
        "stores": smry["auto"],
        "windows": {"baseline": (-0.200, 0.000), "response": (0.0125, 0.100), "stim": (-0.001, 0.005), "overlay": (-0.050, 0.150)},
        "n_trials_indexed": int(len(tt)),
        "n_trials_features": int(len(feat)),
        "X_cols": X_cols,
        "fs": {k: smry["streams"][k]["fs"] for k in smry["streams"]},
        "stim_absmax_median": smry["stim_absmax_median"],
        "used_stim_fallback": bool(used_fallback),
        "I_cols": I_cols,                      # for current vector
        "currents_table_csv": str(I_csv),      # ^^^
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(man, f, indent=2)

    # 7) optional overlay plot
    if save_plot:
        try:
            import matplotlib.pyplot as plt
            pre, post = wins.windows["overlay"]
            auto = auto_select_stores(blk)
            # Prefer our onsets if we computed them earlier
            if HAVE_FALLBACK:
                on_for_plot = get_onsets_with_fallback(blk, auto.epoc, min_epocs=min_epocs_for_fallback)
            else:
                from bme_capstone.dataio.tdt_reader import get_event_onsets
                on_for_plot = get_event_onsets(blk, auto.epoc)

            t, E = epoch_lfp(blk, on_for_plot, auto.lfp, (pre, post))
            fig = plt.figure(figsize=(6,4))
            ax  = fig.add_subplot(111)
            k = min(100, len(E))
            if k > 0:
                ax.plot(t*1e3, E[:k].T, alpha=0.08)
            ax.axvline(0, color="r", lw=1)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("LFP (raw units)")
            ax.set_title(f"{auto.lfp} overlays (n={k}/{len(E)})")
            fig.tight_layout()
            fig.savefig(out_dir / "overlay.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[plot] skipped for {block_dir.name}: {e}")

    # small summary for aggregate
    return {
        "block": block_dir.name,
        "n_trials": int(len(tt)),
        "n_feat": int(len(feat)),
        "X_cols": X_cols,
        "stim_absmax_median": smry["stim_absmax_median"],
        "npz": str(npz_path),
        "features_csv": str(feat_csv),
        "used_fallback": used_fallback,
    }


def main():
    ap = argparse.ArgumentParser(description="Batch TDT → per-block dataset builder")
    ap.add_argument("--root", type=str,
                    default=str(ROOT / "data" / "raw" / "Exp_1"),
                    help="Folder that contains ExperimentBL-*")
    ap.add_argument("--pattern", type=str, default="ExperimentBL-*",
                    help="Glob pattern for blocks")
    ap.add_argument("--save-plot", action="store_true",
                    help="Save overlay.png per block")
    ap.add_argument("--aggregate", action="store_true",
                    help="Write session-level aggregate to data/processed/<session>")
    ap.add_argument("--min-epocs", type=int, default=10,
                    help="Epoc count below which stim-fallback is used (if available)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip block if manifest.json already exists in output")
    ap.add_argument("--stim-only", action="store_true",
                help="Aggregate only trials/blocks with stimulation ON")
    ap.add_argument("--stim-thresh", type=float, default=0.0,
                help="Threshold on stim_absmax_median to consider a block stim-ON")

    args = ap.parse_args()
    root = Path(args.root).resolve()
    session_name = root.name                    # e.g., "Exp_1"

    interim_base = ROOT / "data" / "interim" / session_name
    processed_dir = ROOT / "data" / "processed" / session_name
    interim_base.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    blocks = find_blocks(root, pattern=args.pattern)
    if not blocks:
        print(f"[!] No blocks found under: {root}")
        return

    print(f"[build] Found {len(blocks)} blocks under {root}")
    per_block_stats = []

    for b in blocks:
        out_dir = interim_base / b.name
        if args.skip_existing and (out_dir / "manifest.json").exists():
            print(f"[skip] {b.name} (manifest exists)")
            continue

        print(f"[build] {b.name} → {out_dir}")
        stats = process_block(b, out_dir, save_plot=args.save_plot, min_epocs_for_fallback=args.min_epocs)
        per_block_stats.append(stats)

        # aggregate (optional)
    # ---------------------------------------------------------------
    # If we're skipping existing blocks, per_block_stats will be empty.
    # In that case, rebuild it from the saved NPZ + manifest files.
    # ---------------------------------------------------------------
    if args.aggregate and not per_block_stats:
        print("[aggregate] Collecting existing per-block stats (skip-existing active).")
        for b in blocks:
            out_dir = interim_base / b.name
            npz_path = out_dir / "dataset_tensors_v0.npz"
            feat_csv = out_dir / "features.csv"
            man_path = out_dir / "manifest.json"
            if npz_path.exists():
                d = np.load(npz_path, allow_pickle=True)
                stim_med = 0.0
                if man_path.exists():
                    try:
                        with open(man_path, "r") as f:
                            man = json.load(f)
                        stim_med = float(man.get("stim_absmax_median", 0.0))
                    except Exception:
                        pass
                per_block_stats.append({
                    "block": b.name,
                    "X_cols": list(d["X_cols"]),
                    "npz": str(npz_path),
                    "features_csv": str(feat_csv),
                    "n_trials": int(len(d["y"])),
                    "stim_absmax_median": stim_med,
                    "used_fallback": False,
                })


    # ---------------------------------------------------------------
    # Aggregate across all blocks
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
# Aggregate across all blocks
# ---------------------------------------------------------------
    if args.aggregate and per_block_stats:
        processed_dir.mkdir(parents=True, exist_ok=True)

        if args.stim_only:
            # 1) keep only blocks with stim ON by manifest metric
            targ_cols = ["stim_abs_max", "stim_rms", "lfp_baseline_rms"]
            sel = [s for s in per_block_stats if s.get("stim_absmax_median", 0.0) > args.stim_thresh]
            if not sel:
                print("[aggregate] No stim-ON blocks matched the threshold. Nothing to write.")
            else:
                print(f"[aggregate] Stim-only: using {len(sel)} blocks (threshold={args.stim_thresh}).")
                X_all, y_all, cfg_all = [], [], []
                kept_blocks = 0
                for s in sel:
                    feat = pd.read_csv(s["features_csv"])
                    # drop trials without stim signal (defensive: keep only positive stim_abs_max)
                    if "stim_abs_max" not in feat.columns or "stim_rms" not in feat.columns:
                        continue
                    feat = feat.copy()
                    m = (feat["stim_abs_max"].fillna(0).to_numpy() > 0)
                    feat = feat.loc[m]
                    if len(feat) == 0:
                        continue
                    X, y, cfg, X_cols = build_tensors_v0(feat, input_cols=targ_cols)
                    if X.size == 0:
                        continue
                    X_all.append(X); y_all.append(y); cfg_all.append(cfg)
                    kept_blocks += 1

                if kept_blocks == 0:
                    print("[aggregate] Stim-only: no usable trials after filtering. Nothing to write.")
                else:
                    X = np.vstack(X_all)
                    y = np.vstack(y_all)
                    cfg = np.concatenate(cfg_all)
                    np.savez_compressed(
                        processed_dir / "dataset_tensors_v1_stim.npz",
                        X=X, y=y, cfg=cfg, X_cols=np.array(targ_cols, dtype=object)
                    )
                    pd.DataFrame({"cfg": cfg.reshape(-1)}).to_csv(processed_dir / "meta_cfg_stim.csv", index=False)
                    print(f"[aggregate] wrote tensors → {processed_dir/'dataset_tensors_v1_stim.npz'}  X={X.shape}")

        else:
            # original behavior: try to stack directly; if columns differ, fall back to baseline-only
            cols_sets = {tuple(s["X_cols"]) for s in per_block_stats}
            if len(cols_sets) > 1:
                print("[aggregate] Different X_cols across blocks; falling back to baseline-only aggregation.")
                X_all, y_all, cfg_all = [], [], []
                for s in per_block_stats:
                    feat = pd.read_csv(s["features_csv"])
                    X, y, cfg, X_cols = build_tensors_v0(feat, input_cols=["lfp_baseline_rms"])
                    X_all.append(X); y_all.append(y); cfg_all.append(cfg)
                X = np.vstack(X_all)
                y = np.vstack(y_all)
                cfg = np.concatenate(cfg_all)
                agg_cols = ["lfp_baseline_rms"]
            else:
                X_all, y_all, cfg_all = [], [], []
                agg_cols = list(cols_sets.pop())
                for s in per_block_stats:
                    d = np.load(s["npz"], allow_pickle=True)
                    X_all.append(d["X"]); y_all.append(d["y"]); cfg_all.append(d["cfg"])
                X = np.vstack(X_all)
                y = np.vstack(y_all)
                cfg = np.concatenate(cfg_all)

            if not args.stim_only:
                np.savez_compressed(
                    processed_dir / "dataset_tensors_v0_all.npz",
                    X=X, y=y, cfg=cfg, X_cols=np.array(agg_cols, dtype=object)
                )
                pd.DataFrame({"cfg": cfg.reshape(-1)}).to_csv(processed_dir / "meta_cfg.csv", index=False)
                print(f"[aggregate] wrote tensors → {processed_dir/'dataset_tensors_v0_all.npz'}")

    print("[done] per-block outputs in:", interim_base)
    if args.aggregate:
        print("[done] aggregate outputs in:", processed_dir)



if __name__ == "__main__":
    main()
