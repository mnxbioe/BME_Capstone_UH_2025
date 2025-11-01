#!/usr/bin/env python
"""Run physics-relevant verification for Tower A v2 (fast, minimal).

For the current single-contact workflow, we check:
- Net-current conservation
- Boundary flux residuals
- Interior PDE residuals
Superposition is reported only when multiple contacts are trained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Disable torch.compile/Triton requirement (fallback to eager)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Ensure `src/` is on sys.path when running directly from the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bme_capstone.tower_a_v2 import geometry as geo_mod
from bme_capstone.tower_a_v2.pinn_field import TowerABasisTrainer
from bme_capstone.tower_a_v2.verify.physics import (
    check_net_current_conservation,
    pde_residual_summary,
    boundary_residual_summary,
)
from bme_capstone.tower_a_v2.runlog import start_run

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tower A v2 physics verification")
    p.add_argument("--contact", type=str, default="E01", help="Contact name to verify")
    p.add_argument("--current", type=float, default=1e-6, help="Basis current used during training (A)")
    p.add_argument("--geometry", type=str, default="single_contact_geometry", help="Geometry builder in tower_a_v2.geometry")
    p.add_argument("--preset", type=str, default="smoke", choices=["smoke", "default"], help="Preset")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator (cpu, gpu, auto)")
    p.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    p.add_argument("--max-epochs", type=int, default=None, help="Override preset max epochs")
    p.add_argument("--interior-points", type=int, default=None, help="Override interior collocation points")
    p.add_argument("--contact-points", type=int, default=None, help="Override contact surface points")
    p.add_argument("--outer-points", type=int, default=None, help="Override outer surface points")
    p.add_argument("--shank-points", type=int, default=None, help="Override shank surface points")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build geometry
    if not hasattr(geo_mod, args.geometry):
        raise SystemExit(f"Unknown geometry builder {args.geometry!r} in tower_a_v2.geometry")
    geometry_builder = getattr(geo_mod, args.geometry)
    try:
        geometry = geometry_builder(contact_name=args.contact)
    except TypeError:
        geometry = geometry_builder()
    if args.contact not in {p.name for p in geometry.contacts}:
        raise SystemExit(f"Contact {args.contact!r} not present in geometry {args.geometry!r}")

    conductivity = 0.0002  # S/mm baseline

    # Train single contact basis
    trainer = TowerABasisTrainer(geometry=geometry, conductivity=conductivity)
    if args.preset == "default":
        dkwargs = {
            "interior_points": 200_000,
            "contact_points": 32_768,
            "shank_points": 16_384,
            "outer_points": 16_384,
        }
    else:
        dkwargs = {"interior_points": 5_000, "contact_points": 2_048, "shank_points": 1_024, "outer_points": 1_024}
    if args.interior_points is not None:
        dkwargs["interior_points"] = int(args.interior_points)
    if args.contact_points is not None:
        dkwargs["contact_points"] = int(args.contact_points)
    if args.outer_points is not None:
        dkwargs["outer_points"] = int(args.outer_points)
    if args.shank_points is not None:
        dkwargs["shank_points"] = int(args.shank_points)

    max_epochs = args.max_epochs if args.max_epochs is not None else 300
    if args.preset == "default" and args.max_epochs is None:
        max_epochs = 2000
    tkwargs = {
        "max_epochs": int(max_epochs),
        "enable_model_summary": False,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }

    basis_result = trainer.train_basis(
        args.contact,
        current=args.current,
        discretisation_kwargs=dkwargs,
        trainer_kwargs=tkwargs,
    )

    # Superposition exposed only when multi-contact support is reintroduced
    superposition_rel_error = None

    # Net-current conservation (sum of contact fluxes)
    solvers = {basis_result.contact_name: basis_result.solver}
    conservation = check_net_current_conservation(geometry, solvers, conductivity, n_points=1024)

    # Residual summaries (interior PDE and boundary flux conditions)
    contact_currents = {basis_result.contact_name: args.current}
    pde_summary = pde_residual_summary(geometry, basis_result.solver, conductivity, n_points=2000)
    bc_summary = boundary_residual_summary(geometry, basis_result.solver, conductivity, contact_currents, n_points=1024)

    report = {
        "superposition_rel_error": superposition_rel_error,
        "conservation": conservation,
        "pde_residual": pde_summary,
        "boundary_residuals": bc_summary,
        "preset": args.preset,
        "seed": args.seed,
        "contact": args.contact,
        "geometry_fn": args.geometry,
        "current_A": args.current,
    }
    run_dir = start_run(slug=f"verify_{args.preset}", config={"script": "a2_verify", **report})
    out = Path(run_dir) / "verify"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Smoke health snapshot (single epoch-style JSON)
    health_dir = Path(run_dir) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    with (health_dir / "epoch_0001.json").open("w", encoding="utf-8") as f:
        json.dump({
            "superposition_rel_error": superposition_rel_error,
            "net_contact_current_A": conservation.get("net_contact_current_A", 0.0),
            "pde_residual_median": report["pde_residual"]["median"],
        }, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()
