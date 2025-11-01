# scripts/qa_quickplots.py
# Make quick sanity plots from INTERIM features and/or PROCESSED tensors.

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_hist(ax, x, title, xlabel):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    ax.hist(x, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("count")

def plot_scatter(ax, x, y, title, xlabel, ylabel, alpha=0.2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[m], y[m], '.', alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

def load_all_features(session: str) -> pd.DataFrame:
    """Concatenate all features.csv under data/interim/<session>/*/"""
    base = ROOT / "data" / "interim" / session
    rows = []
    for d in sorted(base.glob("ExperimentBL-*")):
        fcsv = d / "features.csv"
        if fcsv.exists():
            df = pd.read_csv(fcsv)
            df["block"] = d.name
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

def load_block_counts(session: str) -> pd.DataFrame:
    base = ROOT / "data" / "interim" / session
    rows = []
    for d in sorted(base.glob("ExperimentBL-*")):
        m = d / "manifest.json"
        if m.exists():
            with open(m, "r") as f:
                man = json.load(f)
            rows.append(dict(block=d.name,
                             n_trials_indexed=man.get("n_trials_indexed", 0),
                             n_trials_features=man.get("n_trials_features", 0),
                             stim_absmax_median=man.get("stim_absmax_median", 0.0),
                             X_cols=",".join(man.get("X_cols", []))))
    return pd.DataFrame(rows)

def plots_from_features(session: str, outdir: Path):
    F = load_all_features(session)
    if F.empty:
        print("[qa] no features found under data/interim/{}/".format(session))
        return

    # 1) per-block counts
    counts = load_block_counts(session)
    if not counts.empty:
        counts.sort_values("block").to_csv(outdir / "per_block_counts.csv", index=False)

    # 2) histograms
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    plot_hist(ax1, F.get("lfp_baseline_rms", []), "LFP baseline RMS", "RMS (a.u.)")
    plot_hist(ax2, F.get("lfp_response_rms", []), "LFP response RMS", "RMS (a.u.)")

    # 3) baseline vs response scatter
    ax3 = fig.add_subplot(233)
    if "lfp_baseline_rms" in F and "lfp_response_rms" in F:
        plot_scatter(ax3,
                     F["lfp_baseline_rms"], F["lfp_response_rms"],
                     "Response vs Baseline RMS",
                     "baseline RMS", "response RMS")

    # 4) stim histograms & scatter (non-zero only)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    if "stim_abs_max" in F and "stim_rms" in F:
        nz = (F["stim_abs_max"].fillna(0).to_numpy() != 0) | (F["stim_rms"].fillna(0).to_numpy() != 0)
        Fz = F[nz]
        if not Fz.empty:
            plot_hist(ax4, Fz["stim_abs_max"], "Stim |abs|max (non-zero)", "amplitude (a.u.)")
            plot_hist(ax5, Fz["stim_rms"], "Stim RMS (non-zero)", "RMS (a.u.)")
            if "lfp_response_rms" in Fz:
                plot_scatter(ax6, Fz["stim_abs_max"], Fz["lfp_response_rms"],
                             "Stim |abs|max vs LFP response RMS",
                             "stim |abs|max", "response RMS")
    fig.suptitle(f"[features] session={session}")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outdir / "features_overview.png", dpi=150)
    plt.close(fig)
    print(f"[qa] wrote {outdir/'features_overview.png'}")

def plots_from_npz(npz_path: Path, outdir: Path):
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    X_cols = list(d["X_cols"])
    print(f"[qa] NPZ: {npz_path.name}  X={X.shape}  y={y.shape}  X_cols={X_cols}")

    # column-wise histograms
    n = X.shape[1]
    fig = plt.figure(figsize=(4*n, 3))
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        plot_hist(ax, X[:, i], f"{X_cols[i]} (X[:,{i}])", X_cols[i])
    fig.suptitle(npz_path.name)
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(outdir / (npz_path.stem + "_X_hists.png"), dpi=150)
    plt.close(fig)

    # simple pairwise scatters (if small number of cols)
    if n <= 4:
        fig = plt.figure(figsize=(4*n, 4*n))
        k = 1
        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(n, n, k); k += 1
                if i == j:
                    plot_hist(ax, X[:, i], X_cols[i], X_cols[i])
                else:
                    plot_scatter(ax, X[:, j], X[:, i], f"{X_cols[j]} → {X_cols[i]}", X_cols[j], X_cols[i], alpha=0.1)
        fig.suptitle(npz_path.name + " (pairwise)")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(outdir / (npz_path.stem + "_X_pairwise.png"), dpi=150)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Quick sanity plots from interim features / processed tensors")
    ap.add_argument("--session", default="Exp_1", help="session folder under data/")
    ap.add_argument("--npz", default="", help="optional path to a processed npz; if empty, auto-pick")
    args = ap.parse_args()

    session = args.session
    outdir = _mkdir(ROOT / "data" / "processed" / session / "qa")

    # 1) features (interim)
    plots_from_features(session, outdir)

    # 2) processed npz: prefer v1 (stim-only) else v0_all (baseline)
    if args.npz:
        npz = Path(args.npz)
    else:
        pdir = ROOT / "data" / "processed" / session
        cand = [pdir / "dataset_tensors_v1_stim.npz",
                pdir / "dataset_tensors_v0_all.npz"]
        npz = next((p for p in cand if p.exists()), None)

    if npz and npz.exists():
        plots_from_npz(npz, outdir)
    else:
        print("[qa] no processed npz found; skip tensors plots")

if __name__ == "__main__":
    main()
