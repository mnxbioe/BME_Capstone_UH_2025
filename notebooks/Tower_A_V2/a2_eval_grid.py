#!/usr/bin/env python
"""Evaluate phi/E on a Cartesian grid for interpretability (Tower A v2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure `src/` is on sys.path when running directly from the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bme_capstone.tower_a_v2 import geometry as geo_mod
from bme_capstone.tower_a_v2.runlog import start_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate phi/E on a grid (Tower A v2)")
    p.add_argument("--spacing", type=float, default=1.0, help="Grid spacing (mm)")
    p.add_argument("--padding", type=float, default=0.0, help="Padding around volume (mm)")
    p.add_argument("--currents", type=float, nargs="+", default=[1.0], help="Currents for ordered contacts")
    p.add_argument("--geometry", type=str, default="single_contact_reference", help="Geometry builder in tower_a_v2.geometry")
    p.add_argument("--contact", type=str, default="E0", help="Contact order for the provided currents")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not hasattr(geo_mod, args.geometry):
        raise SystemExit(f"Unknown geometry builder {args.geometry!r} in tower_a_v2.geometry")
    geometry_builder = getattr(geo_mod, args.geometry)
    try:
        geometry = geometry_builder(contact_name=args.contact)
    except TypeError:
        geometry = geometry_builder()
    conductivity = 0.0002

    # Placeholder: assume pre-trained basis are loaded elsewhere.
    # For scaffolding: create a dummy evaluator that expects basis dict entries.
    print("This script expects pre-trained basis models and currents to be provided. Stub only.")

    cfg = {
        "script": "a2_eval_grid",
        "spacing_mm": args.spacing,
        "padding_mm": args.padding,
        "currents": args.currents,
        "geometry_fn": args.geometry,
        "contact_order": [args.contact],
    }
    run_dir = start_run(slug="eval_grid", config=cfg)
    out_dir = Path(run_dir) / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Placeholder: load models and evaluate grids here.\n")
    print(f"Scaffold created at: {out_dir}")


if __name__ == "__main__":
    main()
