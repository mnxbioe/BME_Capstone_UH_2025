"""Deterministic integration test for Tower A PINN fields.

This script trains one or more current-injection patterns inside an
8 mm cubic domain and performs a handful of sanity checks:

* geometry/contact summaries so electrode locations are explicit,
* net-current validation against the physics module tolerance,
* PDE residual sampling,
* optional superposition check (left + right ~= both),
* optional plotting of slice data.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pina import LabelTensor
from pina.operator import div, grad

torch.set_float32_matmul_precision("high")

# Ensure ``src`` can be imported when the script is launched from ``scripts``.
_SRC_ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC_ROOT not in sys.path:
    sys.path.append(_SRC_ROOT)

from bme_capstone.tower_a.geometry import Box3D, PlanePatch, TowerAGeometry
from bme_capstone.tower_a.pinn_field import NET_CURRENT_TOL, TowerABasisTrainer

DEFAULT_SEED = 2025
DEFAULT_SLICE_LIMITS = (-2.0, 2.0)
DEFAULT_SLICE_Z = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tower A PINN integration test with reproducible checks."
    )
    parser.add_argument(
        "--pattern",
        choices=["left_only", "right_only", "both", "all"],
        default="all",
        help="Choose a single electrode configuration or run the trio (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--max-epochs", type=int, default=600, help="Lightning max_epochs setting."
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help='Lightning accelerator argument (e.g., "gpu", "cpu", "auto").',
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices.")
    parser.add_argument(
        "--interior-points", type=int, default=40_000, help="Interior sample count."
    )
    parser.add_argument(
        "--contact-points", type=int, default=6_000, help="Boundary sample count/contact."
    )
    parser.add_argument(
        "--outer-points", type=int, default=4_000, help="Outer boundary sample count."
    )
    parser.add_argument(
        "--pde-samples",
        type=int,
        default=4_000,
        help="Number of random interior points for PDE residual estimates.",
    )
    parser.add_argument(
        "--slice-limits",
        type=float,
        nargs=2,
        default=DEFAULT_SLICE_LIMITS,
        metavar=("XMIN", "XMAX"),
        help="Lateral limits (mm) for slice visualisation grids.",
    )
    parser.add_argument(
        "--slice-n",
        type=int,
        default=121,
        help="Number of points per axis for the slice grid (>= 3).",
    )
    parser.add_argument(
        "--slice-z",
        type=float,
        default=DEFAULT_SLICE_Z,
        help="Z location (mm) of the evaluation slice.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib import/usage (useful on headless machines).",
    )
    parser.add_argument(
        "--matplotlib-backend",
        default="TkAgg",
        help='Backend used if plots are enabled (e.g., "TkAgg", "Agg", "QtAgg").',
    )
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_geometry() -> TowerAGeometry:
    """Return the standard Tower A cube with two square contacts on z=0."""
    volume = Box3D(x=(-4, 4), y=(-4, 4), z=(-4, 4))
    half_size = 0.2  # mm, giving a 0.4 mm x 0.4 mm patch
    contacts = [
        PlanePatch(
            name="E_left",
            axis="z",
            value=0.0,
            span={"x": (-0.8 - half_size, -0.8 + half_size), "y": (-half_size, half_size)},
            normal_sign=+1,
            kind="contact",
        ),
        PlanePatch(
            name="E_right",
            axis="z",
            value=0.0,
            span={"x": (+0.8 - half_size, +0.8 + half_size), "y": (-half_size, half_size)},
            normal_sign=+1,
            kind="contact",
        ),
    ]
    outers = [
        PlanePatch(
            name="z_top",
            axis="z",
            value=+4.0,
            span={"x": (-4, 4), "y": (-4, 4)},
            normal_sign=+1,
            kind="outer",
        ),
        PlanePatch(
            name="z_bottom",
            axis="z",
            value=-4.0,
            span={"x": (-4, 4), "y": (-4, 4)},
            normal_sign=-1,
            kind="outer",
        ),
    ]
    return TowerAGeometry(volume=volume, contacts=contacts, shanks=[], outers=outers)


def default_outer_bc() -> Dict[str, Dict[str, float]]:
    return {
        "z_top": {"type": "dirichlet", "value": 0.0},
        "z_bottom": {"type": "dirichlet", "value": 0.0},
    }


def conductivity_s_mm() -> float:
    """Convert 0.3 S/m to S/mm."""
    return 0.3 / 1_000.0


def micro_amp(value_micro_amp: float) -> float:
    """Convert micro-ampere input to ampere."""
    return value_micro_amp * 1e-6


def standard_patterns(current_amp: float) -> Dict[str, Dict[str, float]]:
    """Return the nominal three current patterns."""
    return {
        "left_only": {"E_left": current_amp, "E_right": 0.0},
        "right_only": {"E_left": 0.0, "E_right": -current_amp},
        "both": {"E_left": current_amp, "E_right": -current_amp},
    }


def select_patterns(
    all_patterns: Mapping[str, Dict[str, float]], selector: str
) -> Dict[str, Dict[str, float]]:
    if selector == "all":
        return dict(all_patterns)
    return {selector: dict(all_patterns[selector])}


def describe_geometry(geometry: TowerAGeometry) -> None:
    vol = geometry.volume
    print(
        f"Volume extents (mm): x[{vol.x[0]:+.1f}, {vol.x[1]:+.1f}] "
        f"y[{vol.y[0]:+.1f}, {vol.y[1]:+.1f}] "
        f"z[{vol.z[0]:+.1f}, {vol.z[1]:+.1f}]"
    )
    print("Contacts:")
    for patch in geometry.contacts:
        tangential_axes = [ax for ax in ("x", "y", "z") if ax != patch.axis]
        spans = {ax: patch.span[ax] for ax in tangential_axes}
        centers = {
            patch.axis: patch.value,
            tangential_axes[0]: 0.5 * (spans[tangential_axes[0]][0] + spans[tangential_axes[0]][1]),
            tangential_axes[1]: 0.5 * (spans[tangential_axes[1]][0] + spans[tangential_axes[1]][1]),
        }
        lengths = {
            ax: spans[ax][1] - spans[ax][0]
            for ax in tangential_axes
        }
        area_mm2 = lengths[tangential_axes[0]] * lengths[tangential_axes[1]]
        print(
            f"  {patch.name:8s} axis={patch.axis} z={patch.value:+.3f} mm "
            f"normal={patch.normal_sign:+d}"
        )
        print(
            f"    spans: {tangential_axes[0]}[{spans[tangential_axes[0]][0]:+.3f}, "
            f"{spans[tangential_axes[0]][1]:+.3f}] mm | "
            f"{tangential_axes[1]}[{spans[tangential_axes[1]][0]:+.3f}, "
            f"{spans[tangential_axes[1]][1]:+.3f}] mm"
        )
        print(
            f"    centre: ({centers['x']:+.3f}, {centers['y']:+.3f}, {centers['z']:+.3f}) mm "
            f"area={area_mm2 * 1e6:.1f} um^2"
        )


def describe_currents(
    patterns: Mapping[str, Mapping[str, float]], has_dirichlet_ground: bool
) -> None:
    for name, currents in patterns.items():
        total = sum(currents.values())
        status = "ok"
        if not has_dirichlet_ground and abs(total) > NET_CURRENT_TOL:
            status = "invalid (no ground)"
        print(f"Pattern '{name}':")
        for contact, current in currents.items():
            print(f"  {contact:8s} {current:+.3e} A")
        print(f"  net current {total:+.3e} A -> {status}")


def report_contact_conditions(problem, label: str) -> None:
    print(f"\nContact flux targets for pattern '{label}':")
    any_contacts = False
    for key, summary in problem.surface_conditions.items():
        if not key.startswith("contact"):
            continue
        any_contacts = True
        print(f"  {summary.surface.name:10s} ({summary.condition_type}) -> {summary.description}")
    if not any_contacts:
        print("  (no contact conditions registered)")


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_and_train(
    trainer: TowerABasisTrainer,
    currents: Mapping[str, float],
    *,
    outer_bc: Mapping[str, Mapping[str, float]],
    discretisation_kwargs: Dict,
    solver_kwargs: Dict,
    trainer_kwargs: Dict,
):
    problem = trainer.build_problem(contact_currents=currents, outer_bc=outer_bc)
    trainer.discretise(problem, **discretisation_kwargs)
    model = trainer.build_model()
    solver = trainer.solver_cls(problem=problem, model=model, **solver_kwargs)
    trainer_obj = trainer.Trainer(
        solver=solver,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        batch_size=None,
        **trainer_kwargs,
    )
    trainer_obj.train()
    return problem, model, solver, trainer_obj


def eval_phi(model: torch.nn.Module, xs, ys, zs):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    device = _infer_model_device(model)
    pts = torch.tensor(
        np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        out = model(pts).cpu().numpy()
    return X, Y, Z, out.reshape(X.shape)


def pde_residual_mean(problem, model, n: int = 5_000) -> float:
    samples = problem.spatial_domain.sample(n=n, mode="random")
    pts = samples.tensor if isinstance(samples, LabelTensor) else samples
    pts = LabelTensor(pts, labels=["x", "y", "z"])
    pts.requires_grad_(True)

    out = model(pts)
    tensor = out.tensor if isinstance(out, LabelTensor) else out
    phi = LabelTensor(tensor, labels=["phi"])

    grad_phi = grad(phi, pts, components=["phi"], d=["x", "y", "z"])
    flux = float(problem.conductivity) * grad_phi.tensor
    flux_lt = LabelTensor(flux, labels=["J_x", "J_y", "J_z"])
    residual = div(flux_lt, pts, components=flux_lt.labels, d=["x", "y", "z"]).tensor
    return float(residual.abs().mean().cpu())


def compute_superposition_error(phi_left, phi_right, phi_both) -> float:
    numerator = np.mean(np.abs((phi_left + phi_right) - phi_both))
    denom = np.maximum(1e-9, np.mean(np.abs(phi_both)))
    return float(numerator / denom)


def maybe_plot(
    slice_fields: Mapping[str, np.ndarray],
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    backend: str,
) -> None:
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use(backend)
    plt = importlib.import_module("matplotlib.pyplot")

    X, Y, _ = grid
    names = list(slice_fields.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        field = slice_fields[name][:, :, 0]
        im = ax.contourf(X[:, :, 0], Y[:, :, 0], field, levels=20)
        ax.set_title(name)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.figure(figsize=(6, 4))
    y_idx = slice_fields[names[0]].shape[1] // 2
    xs = X[:, 0, 0]
    for name in names:
        plt.plot(xs, slice_fields[name][:, y_idx, 0], label=name)
    plt.title(f"Line cut (y~0, z={grid[2][0,0,0]:+.2f} mm)")
    plt.xlabel("x [mm]")
    plt.ylabel("phi [V]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if args.slice_n < 3:
        raise ValueError("--slice-n must be >= 3.")
    if args.slice_limits[0] >= args.slice_limits[1]:
        raise ValueError("--slice-limits must satisfy XMIN < XMAX.")

    set_deterministic_seed(args.seed)

    geometry = build_geometry()
    outer_bc = default_outer_bc()
    has_dirichlet = any(spec["type"] == "dirichlet" for spec in outer_bc.values())
    describe_geometry(geometry)

    sigma = conductivity_s_mm()
    current = micro_amp(10.0)  # 10 uA
    all_patterns = standard_patterns(current)
    patterns = select_patterns(all_patterns, args.pattern)
    describe_currents(patterns, has_dirichlet_ground=has_dirichlet)

    early_stop = EarlyStopping(
        monitor="pde_loss",
        min_delta=1.0e-8,
        patience=25,
        mode="min",
    )
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "enable_model_summary": False,
        "log_every_n_steps": 50,
        "callbacks": [early_stop],
    }
    discretisation_kwargs = {
        "interior_points": args.interior_points,
        "contact_points": args.contact_points,
        "outer_points": args.outer_points,
        "interior_mode": "latin",
        "boundary_mode": "random",
    }
    solver_kwargs = {"use_lt": True}

    trainer = TowerABasisTrainer(geometry=geometry, conductivity=sigma)

    results = {}
    for name, currents in patterns.items():
        print(f"\n--- Training pattern '{name}' ---")
        problem, model, solver, trainer_obj = build_and_train(
            trainer,
            currents,
            outer_bc=outer_bc,
            discretisation_kwargs=discretisation_kwargs,
            solver_kwargs=solver_kwargs,
            trainer_kwargs=trainer_kwargs,
        )
        results[name] = {
            "problem": problem,
            "model": model,
            "solver": solver,
            "trainer": trainer_obj,
        }
        report_contact_conditions(problem, name)

    if not results:
        print("No patterns were scheduled. Nothing to do.")
        return

    # PDE residual sanity check
    for name, payload in results.items():
        mean_res = pde_residual_mean(payload["problem"], payload["model"], n=args.pde_samples)
        print(f"PDE residual mean |div(sigma grad(phi))| for {name}: {mean_res:.3e}")

    # Evaluate requested slices
    xs = np.linspace(args.slice_limits[0], args.slice_limits[1], args.slice_n)
    ys = np.linspace(args.slice_limits[0], args.slice_limits[1], args.slice_n)
    zs = np.array([args.slice_z])

    grids: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    slice_fields: Dict[str, np.ndarray] = {}
    for name, payload in results.items():
        grid = eval_phi(payload["model"], xs, ys, zs)
        if grids is None:
            grids = grid[:3]
        slice_fields[name] = grid[3]

    if {"left_only", "right_only", "both"}.issubset(results.keys()):
        err = compute_superposition_error(
            slice_fields["left_only"], slice_fields["right_only"], slice_fields["both"]
        )
        print(f"Superposition relative error on z={args.slice_z:+.2f} mm slice: {err:.3e}")
    else:
        print("Skipping superposition check (requires left_only, right_only, and both).")

    if not args.no_plots and slice_fields and grids is not None:
        maybe_plot(slice_fields, grids, backend=args.matplotlib_backend)
    elif args.no_plots:
        print("Plotting disabled (--no-plots).")


if __name__ == "__main__":
    main()
