#!/usr/bin/env python
"""Train a single Tower A v2 basis field with minimal, explicit settings.

Geometry is defined in code. Configuration is snapshot to /runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from pina import LabelTensor
import pyvista as pv
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Disable torch.compile/Triton requirement (fallback to eager)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Ensure `src/` is on sys.path so `bme_capstone` is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bme_capstone.tower_a_v2 import geometry as geo_mod
from bme_capstone.tower_a_v2.pinn_field import TowerABasisTrainer
from bme_capstone.tower_a_v2.runlog import start_run
from bme_capstone.tower_a_v2.visualize import plot_collocation_samples, plot_field_slices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Improve matmul precision on CUDA to leverage tensor cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one Tower A v2 basis field")
    p.add_argument("contact", type=str, nargs="?", default="E01", help="Contact name to drive as basis")
    p.add_argument("--current", type=float, default=1e-6, help="Basis current (A)")
    p.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    p.add_argument(
        "--preset",
        type=str,
        default="smoke",
        choices=["smoke", "default", "thorough"],
        help="Training preset",
    )
    p.add_argument("--geometry", type=str, default="single_contact_geometry", help="Geometry builder in tower_a_v2.geometry")
    p.add_argument("--plot", action="store_true", help="Plot E-field slices and collocation samples after training")
    p.add_argument("--plot-backend", type=str, default=None, help="Matplotlib backend (e.g., 'Agg', 'TkAgg')")
    p.add_argument("--plot-limits", type=float, nargs=2, default=(-2.0, 2.0), help="Field slice limits [min max] in mm")
    p.add_argument("--plot-grid-n", type=int, default=121, help="Number of points per axis for field slices")
    p.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator (cpu, gpu, auto)")
    p.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    p.add_argument("--max-epochs", type=int, default=None, help="Override preset max epochs")
    p.add_argument("--interior-points", type=int, default=None, help="Override interior collocation points")
    p.add_argument("--contact-points", type=int, default=None, help="Override contact surface points")
    p.add_argument("--outer-points", type=int, default=None, help="Override outer surface points")
    p.add_argument("--shank-points", type=int, default=None, help="Override shank surface points")
    return p.parse_args()


def preset_trainer_kwargs(name: str) -> dict:
    if name == "smoke":
        return {"max_epochs": 300, "enable_model_summary": False}
    if name == "default":
        return {"max_epochs": 2000, "enable_model_summary": False}
    if name == "thorough":
        return {"max_epochs": 5000, "enable_model_summary": False}
    return {}


def preset_discretisation(name: str) -> dict:
    if name == "smoke":
        return {"interior_points": 5_000, "contact_points": 2_048, "shank_points": 1_024, "outer_points": 1_024}
    if name == "default":
        return {"interior_points": 200_000, "contact_points": 32_768, "shank_points": 16_384, "outer_points": 16_384}
    if name == "thorough":
        return {"interior_points": 400_000, "contact_points": 65_536, "shank_points": 32_768, "outer_points": 32_768}
    return {}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Build geometry from a named builder in geometry.py
    if not hasattr(geo_mod, args.geometry):
        raise SystemExit(f"Unknown geometry builder {args.geometry!r} in tower_a_v2.geometry")
    geometry_builder = getattr(geo_mod, args.geometry)
    try:
        geometry = geometry_builder(contact_name=args.contact)
    except TypeError:
        geometry = geometry_builder()
    if args.contact not in {p.name for p in geometry.contacts}:
        raise SystemExit(f"Unknown contact {args.contact!r} produced by geometry builder {args.geometry!r}.")

    # Placeholder homogeneous conductivity (S/mm)
    conductivity = 0.0002  # example: 0.2 S/m -> 0.0002 S/mm

    trainer = TowerABasisTrainer(geometry=geometry, conductivity=conductivity)

    discretisation = preset_discretisation(args.preset)
    if args.interior_points is not None:
        discretisation["interior_points"] = int(args.interior_points)
    if args.contact_points is not None:
        discretisation["contact_points"] = int(args.contact_points)
    if args.outer_points is not None:
        discretisation["outer_points"] = int(args.outer_points)
    if args.shank_points is not None:
        discretisation["shank_points"] = int(args.shank_points)

    trainer_kwargs = {
        **{
            **preset_trainer_kwargs(args.preset),
            **({"max_epochs": int(args.max_epochs)} if args.max_epochs is not None else {}),
        },
        "accelerator": args.accelerator,
        "devices": args.devices,
    }

    cfg = {
        "script": "a2_train_basis",
        "preset": args.preset,
        "seed": args.seed,
        "contact": args.contact,
        "current_A": args.current,
        "geometry_fn": args.geometry,
        "conductivity_S_per_mm": conductivity,
        "discretisation": discretisation,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "max_epochs": trainer_kwargs.get("max_epochs"),
    }

    run_dir = start_run(slug=f"{args.contact}_{args.preset}", config=cfg)

    # Train
    result = trainer.train_basis(
        contact_name=args.contact,
        current=args.current,
        trainer_kwargs=trainer_kwargs,
        discretisation_kwargs=discretisation,
    )

    # Save a lightweight checkpoint (torch state_dict)
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), ckpt_dir / f"model_{args.contact}.pt")

    if args.plot:
        # Center slices around the active contact to make visuals clearer
        contact_patch = next(p for p in geometry.contacts if p.name == args.contact)
        cx = sum(contact_patch.span["x"]) * 0.5
        cy = sum(contact_patch.span["y"]) * 0.5
        # Slice slightly inside the domain beneath the contact face
        # --- Multiple depth slices ---
        depth_increments = [0.1, 1.0, 3.0]  # mm above electrode plane
        limits = tuple(sorted(args.plot_limits))

        for dz in depth_increments:
            z_slice = geometry.volume.z[0] + dz
            print(f"Plotting field slice at z = {z_slice:.2f} mm")

            plot_field_slices(
                result.solver,
                geometry,
                x0=cx,
                y0=cy,
                z0=z_slice,
                limits=limits,
                grid_n=args.plot_grid_n,
                backend=args.plot_backend,
            )
            # --- 3D Field Visualization (PyVista) ---
        print("Generating 3D field visualization...")
        from pina import LabelTensor
        import pyvista as pv

        # Helper: compute E = -∇phi
        def compute_electric_field(model: torch.nn.Module, pts: np.ndarray) -> np.ndarray:
            """Compute electric field from a PINA model using autograd."""
            param = next(model.parameters())
            device = param.device
            dtype = param.dtype

            coords_tensor = torch.tensor(pts, dtype=dtype, device=device, requires_grad=True)
            coords = LabelTensor(coords_tensor, labels=["x", "y", "z"])

            outputs = model(coords)
            phi_tensor = (
                outputs.extract(["phi"]).tensor.squeeze(-1)
                if hasattr(outputs, "extract")
                else outputs
            )

            grad = torch.autograd.grad(
                phi_tensor,
                coords_tensor,
                grad_outputs=torch.ones_like(phi_tensor),
                create_graph=False,
                retain_graph=False,
            )[0]
            return (-grad).detach().cpu().numpy()

        # Generate 3D grid
        x = np.linspace(limits[0], limits[1], 41)
        y = np.linspace(limits[0], limits[1], 41)
        z = np.linspace(geometry.volume.z[0], geometry.volume.z[0] + 3.0, 41)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Compute electric field
        E = compute_electric_field(result.solver, pts)
        E_mag = np.linalg.norm(E, axis=1)

                # Create PyVista volume
        grid = pv.StructuredGrid(X, Y, Z)
        vectors = E.reshape(-1, 3)  # ensure contiguous (n_points, 3)
        grid["|E|"] = E_mag  # scalars must be flat and match n_points
        grid["E"] = vectors

        p = pv.Plotter()
        p.add_volume(
        grid, scalars="|E|", cmap="plasma",
        opacity="linear", shade=True, clim=[0, np.max(E_mag)*0.8])

        p.add_arrows(grid.points[::500], vectors[::500], mag=2.0, color="white")
        p.add_axes()
        p.show_grid()
        p.show(title="3D Electric Field Magnitude")
        p.export_html("E_field_3D.html")



    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
