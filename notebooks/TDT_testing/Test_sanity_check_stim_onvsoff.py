# scripts/sanity_check_stim_onvsoff_checked.py
"""
Compare LFP response RMS between stim-ON and stim-OFF trials.
Verifies output location and opens the QA folder automatically.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# -------------------------------------------------------------------
# 1️⃣ Locate project root and data
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "data" / "processed" / "Exp_1" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 2️⃣ Load all feature tables from interim
# -------------------------------------------------------------------
interim_dir = ROOT / "data" / "interim" / "Exp_1"
feature_files = list(interim_dir.rglob("features.csv"))
print(f"[qa] Found {len(feature_files)} feature files")

if not feature_files:
    raise FileNotFoundError("No feature.csv files found in interim/Exp_1")

F = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)
print(f"[qa] Combined feature rows: {len(F)}")

# -------------------------------------------------------------------
# 3️⃣ Define stim-on flag and make boxplot
# -------------------------------------------------------------------
F["stim_on"] = F["stim_abs_max"].fillna(0) > 0

plt.figure(figsize=(5, 4))
plt.boxplot(
    [F.loc[~F.stim_on, "lfp_response_rms"],
     F.loc[F.stim_on, "lfp_response_rms"]],
    tick_labels=["Stim OFF", "Stim ON"],  # updated for Matplotlib ≥3.9
)
plt.ylabel("LFP response RMS (a.u.)")
plt.title("Evoked Response Strength: Stim OFF vs ON")
plt.tight_layout()

# -------------------------------------------------------------------
# 4️⃣ Save figure and confirm
# -------------------------------------------------------------------
outpath = QA_DIR / "dose_response_box.png"
plt.savefig(outpath, dpi=150)
plt.close()

if outpath.exists():
    size_kb = outpath.stat().st_size / 1024
    print(f"[qa] Saved plot → {outpath}")
    print(f"[qa] File size: {size_kb:.1f} KB")
    # Optional: open folder in Explorer
    os.startfile(QA_DIR)
else:
    print(f"[qa] Failed to save plot at {outpath}")
