"""Train single-electrode Tower A and visualise E-field + collocation.

Uses the direct one-electrode geometry builder so parameters are explicit.
Trains a single basis field, then plots 2D electric field slices across
XY, XZ, and YZ planes and a scatter of collocation samples (representative).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

# Ensure src import works when launched from scripts/
_HERE = Path(__file__).resolve().parent
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from bme_capstone.tower_a import TowerABasisTrainer  # type: ignore
from bme_capstone.tower_a.geometry import TowerAGeometry  # type: ignore
from pina import LabelTensor  # type: ignore
from pina.operator import grad  # type: ignore

# Local import within scripts folder
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))
from One_electrode_geometry_builder import build_geometry


def outer_bc_for_single_gauge(geometry: TowerAGeometry) -> Mapping[str, Mapping[str, float]]:
    names = {p.name for p in geometry.outers}
    mapping = {name: {"type": "neumann", "value": 0.0} for name in names}
    # Ensure gauge (if present) is Dirichlet 0 V
    if "gauge" in names:
        mapping["gauge"] = {"type": "dirichlet", "value": 0.0}
    return mapping


def _infer_model_device_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    params = list(model.parameters())
    if params:
        return params[0].device, params[0].dtype
    return torch.device("gpu"), torch.float32


def _grid_xy(geom: TowerAGeometry, xlim: Tuple[float, float], ylim: Tuple[float, float], z: float, n: int):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, z)
    return X, Y, Z


def _grid_xz(geom: TowerAGeometry, xlim: Tuple[float, float], zlim: Tuple[float, float], y: float, n: int):
    xs = np.linspace(xlim[0], xlim[1], n)
    zs = np.linspace(zlim[0], zlim[1], n)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Y = np.full_like(X, y)
    return X, Y, Z


def _grid_yz(geom: TowerAGeometry, ylim: Tuple[float, float], zlim: Tuple[float, float], x: float, n: int):
    ys = np.linspace(ylim[0], ylim[1], n)
    zs = np.linspace(zlim[0], zlim[1], n)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    X = np.full_like(Y, x)
    return X, Y, Z


def _eval_phi_and_E(model: torch.nn.Module, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    device, dtype = _infer_model_device_dtype(model)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), dtype=dtype, device=device)
    pts = LabelTensor(pts, labels=["x", "y", "z"])  # PINA expects labeled tensor
    pts.requires_grad_(True)

    out = model(pts)
    out_t = out.tensor if isinstance(out, LabelTensor) else out
    phi = LabelTensor(out_t, labels=["phi"])  # scalar field
    g = grad(phi, pts, components=["phi"], d=["x", "y", "z"]).tensor  # [N,3]
    Ex = (-g[:, 0]).reshape(X.shape).detach().cpu().numpy()
    Ey = (-g[:, 1]).reshape(X.shape).detach().cpu().numpy()
    Ez = (-g[:, 2]).reshape(X.shape).detach().cpu().numpy()
    Phi = out_t.reshape(X.shape).detach().cpu().numpy()
    return Phi, Ex, Ey, Ez


def _plot_slices(model: torch.nn.Module, geometry: TowerAGeometry, *, x0=0.0, y0=0.0, z0=0.0,
                 limits=(-2.0, 2.0), n=121, backend="TkAgg") -> None:
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    vol = geometry.volume
    xlim = (max(vol.x[0], limits[0]), min(vol.x[1], limits[1]))
    ylim = (max(vol.y[0], limits[0]), min(vol.y[1], limits[1]))
    zlim = (max(vol.z[0], limits[0]), min(vol.z[1], limits[1]))

    # XY @ z0
    Xxy, Yxy, Zxy = _grid_xy(geometry, xlim, ylim, z0, n)
    _, Ex_xy, Ey_xy, _ = _eval_phi_and_E(model, Xxy, Yxy, Zxy)
    mag_xy = np.sqrt(Ex_xy**2 + Ey_xy**2)

    # XZ @ y0
    Xxz, Yxz, Zxz = _grid_xz(geometry, xlim, zlim, y0, n)
    _, Ex_xz, _, Ez_xz = _eval_phi_and_E(model, Xxz, Yxz, Zxz)
    mag_xz = np.sqrt(Ex_xz**2 + Ez_xz**2)

    # YZ @ x0
    Xyz, Yyz, Zyz = _grid_yz(geometry, ylim, zlim, x0, n)
    _, _, Ey_yz, Ez_yz = _eval_phi_and_E(model, Xyz, Yyz, Zyz)
    mag_yz = np.sqrt(Ey_yz**2 + Ez_yz**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(mag_xy.T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='viridis')
    axes[0].quiver(Xxy[::6, ::6], Yxy[::6, ::6], Ex_xy[::6, ::6], Ey_xy[::6, ::6], color='w', alpha=0.7)
    axes[0].set_title(f"E magnitude (XY @ z={z0:+.2f} mm)")
    axes[0].set_xlabel('x [mm]'); axes[0].set_ylabel('y [mm]')

    im1 = axes[1].imshow(mag_xz.T, origin='lower', extent=[xlim[0], xlim[1], zlim[0], zlim[1]], cmap='viridis')
    axes[1].quiver(Xxz[::6, ::6], Zxz[::6, ::6], Ex_xz[::6, ::6], Ez_xz[::6, ::6], color='w', alpha=0.7)
    axes[1].set_title(f"E magnitude (XZ @ y={y0:+.2f} mm)")
    axes[1].set_xlabel('x [mm]'); axes[1].set_ylabel('z [mm]')

    im2 = axes[2].imshow(mag_yz.T, origin='lower', extent=[ylim[0], ylim[1], zlim[0], zlim[1]], cmap='viridis')
    axes[2].quiver(Yyz[::6, ::6], Zyz[::6, ::6], Ey_yz[::6, ::6], Ez_yz[::6, ::6], color='w', alpha=0.7)
    axes[2].set_title(f"E magnitude (YZ @ x={x0:+.2f} mm)")
    axes[2].set_xlabel('y [mm]'); axes[2].set_ylabel('z [mm]')

    cbar = fig.colorbar(im0, ax=axes[0]); cbar.set_label('|E| [a.u.]')
    fig.colorbar(im1, ax=axes[1]); fig.colorbar(im2, ax=axes[2])
    plt.show()


def _plot_collocation(problem, *, interior_points: int, contact_points: int, outer_points: int, backend="TkAgg") -> None:
    # Representative scatter: resample domains similarly to training discretisation
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')

    # Interior
    interior_name = problem.geometry.interior_name
    inter = problem.domains[interior_name].sample(n=interior_points, mode="latin")
    P = inter.tensor if isinstance(inter, LabelTensor) else inter
    P = P.cpu().numpy()
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.2, label='interior')

    # Contacts
    for patch in problem.geometry.contacts:
        dom = problem.domains[patch.domain_name("contact")]
        samp = dom.sample(n=contact_points, mode="random")
        Q = samp.tensor if isinstance(samp, LabelTensor) else samp
        Q = Q.cpu().numpy()
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=3, alpha=0.6, label=f'contact:{patch.name}')

    # Outers
    for patch in problem.geometry.outers:
        dom = problem.domains[patch.domain_name("outer")]
        samp = dom.sample(n=outer_points, mode="random")
        R = samp.tensor if isinstance(samp, LabelTensor) else samp
        R = R.cpu().numpy()
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], s=1, alpha=0.2, label=f'outer:{patch.name}')

    ax.set_title('Representative collocation samples')
    # Keep legend light
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right', fontsize=8)
    plt.tight_layout(); plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train single-electrode PINN and visualise E-field + samples")
    p.add_argument('--current-uA', type=float, default=10.0, help='Contact current (microampere)')
    p.add_argument('--sigma-S-per-m', type=float, default=0.3, help='Conductivity (S/m)')
    p.add_argument('--max-epochs', type=int, default=600)
    p.add_argument('--accelerator', default='auto')
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--interior-points', type=int, default=40_000)
    p.add_argument('--contact-points', type=int, default=6_000)
    p.add_argument('--outer-points', type=int, default=4_000)
    p.add_argument('--grid-limits', type=float, nargs=2, default=(-2.0, 2.0))
    p.add_argument('--grid-n', type=int, default=121)
    p.add_argument('--x0', type=float, default=0.0)
    p.add_argument('--y0', type=float, default=0.0)
    p.add_argument('--z0', type=float, default=0.0)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--matplotlib-backend', default='TkAgg')
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # 1) Geometry
    geometry = build_geometry()

    # 2) Physics and trainer
    sigma = float(args.sigma_S_per_m) / 1000.0  # convert S/m -> S/mm
    current_A = args.current_uA * 1e-6
    trainer = TowerABasisTrainer(geometry=geometry, conductivity=sigma)

    # Make outer BC: all Neumann 0 except gauge Dirichlet 0
    outer_bc = outer_bc_for_single_gauge(geometry)

    result = trainer.train_basis(
        contact_name=geometry.contacts[0].name,
        current=current_A,
        outer_bc=outer_bc,
        discretisation_kwargs={
            'interior_points': args.interior_points,
            'contact_points': args.contact_points,
            'outer_points': args.outer_points,
            'interior_mode': 'latin',
            'boundary_mode': 'random',
        },
        trainer_kwargs={
            'max_epochs': args.max_epochs,
            'accelerator': args.accelerator,
            'devices': args.devices,
            'enable_model_summary': False,
            'enable_checkpointing': False,
            'logger': False,
        },
        solver_kwargs={'use_lt': True},
    )

    print("Training complete.")
    # Save model weights and geometry
    save_dir = Path(r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\scripts\Tower_A_TestV1")
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(result.model.state_dict(), save_dir / "towerA_single_electrode.pt")
    torch.save(geometry, save_dir / "geometry_obj.pt")

    print(f"Saved model and geometry to {save_dir}")


    if not args.no_plots:
        _plot_slices(result.model, geometry, x0=args.x0, y0=args.y0, z0=args.z0,
                     limits=tuple(args.grid_limits), n=args.grid_n, backend=args.matplotlib_backend)
        _plot_collocation(result.problem,
                          interior_points=min(args.interior_points, 5000),
                          contact_points=min(args.contact_points, 2000),
                          outer_points=min(args.outer_points, 2000),
                          backend=args.matplotlib_backend)


if __name__ == '__main__':
    main()

