"""Visualization helpers for Tower A v2.

These routines are optional and import ``matplotlib`` lazily so the core
package remains lightweight. They provide:

- electric-field slice plots through the trained basis field
- representative collocation sample scatter plots
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch

from pina import LabelTensor
from pina.operator import grad

from .geometry import TowerAGeometry
from .pinn_field import DEFAULT_AXES, TowerALaplaceProblem, SolverType


def _ensure_matplotlib(backend: str | None = None):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    if backend:
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    return plt


def _infer_device_dtype(module: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    params = list(module.parameters())
    if params:
        return params[0].device, params[0].dtype
    return torch.device("cpu"), torch.float32


def _label_tensor_from_arrays(*arrays: np.ndarray, device: torch.device, dtype: torch.dtype) -> LabelTensor:
    stacked = np.stack(arrays, axis=1)
    tensor = torch.tensor(stacked, dtype=dtype, device=device)
    lt = LabelTensor(tensor, list(DEFAULT_AXES))
    lt.requires_grad_(True)
    return lt


def _evaluate_field(solver: SolverType, coords: LabelTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    solver.eval()
    output = solver.forward(coords).extract(["phi"])
    grad_phi = grad(output, coords, components=["phi"], d=list(DEFAULT_AXES)).tensor
    return output.tensor.squeeze(-1).detach(), (-grad_phi).detach()


def plot_field_slices(
    solver: SolverType,
    geometry: TowerAGeometry,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    limits: Tuple[float, float] = (-2.0, 2.0),
    grid_n: int = 121,
    backend: str | None = None,
) -> None:
    """Plot |E| slices on XY, XZ, YZ planes with quiver overlays."""

    plt = _ensure_matplotlib(backend)

    vol = geometry.volume
    lo, hi = limits
    xlim = (max(vol.x[0], lo), min(vol.x[1], hi))
    ylim = (max(vol.y[0], lo), min(vol.y[1], hi))
    zlim = (max(vol.z[0], lo), min(vol.z[1], hi))

    device, dtype = _infer_device_dtype(solver.model)

    def _grid(a0: float, a1: float, b0: float, b1: float) -> Tuple[np.ndarray, np.ndarray]:
        axis_a = np.linspace(a0, a1, grid_n)
        axis_b = np.linspace(b0, b1, grid_n)
        return np.meshgrid(axis_a, axis_b, indexing="ij")

    # XY slice at z0
    Xxy, Yxy = _grid(*xlim, *ylim)
    Zxy = np.full_like(Xxy, np.clip(z0, zlim[0], zlim[1]))
    coords_xy = _label_tensor_from_arrays(Xxy.ravel(), Yxy.ravel(), Zxy.ravel(), device=device, dtype=dtype)
    phi_xy, field_xy = _evaluate_field(solver, coords_xy)
    field_xy_np = field_xy.cpu().numpy()
    Ex_xy = field_xy_np[:, 0].reshape(Xxy.shape)
    Ey_xy = field_xy_np[:, 1].reshape(Xxy.shape)
    mag_xy = np.linalg.norm(field_xy_np[:, :2], axis=1).reshape(Xxy.shape)

    # XZ slice at y0
    Xxz, Zxz = _grid(*xlim, *zlim)
    Yxz = np.full_like(Xxz, np.clip(y0, ylim[0], ylim[1]))
    coords_xz = _label_tensor_from_arrays(Xxz.ravel(), Yxz.ravel(), Zxz.ravel(), device=device, dtype=dtype)
    _, field_xz = _evaluate_field(solver, coords_xz)
    field_xz_np = field_xz.cpu().numpy()
    Ex_xz = field_xz_np[:, 0].reshape(Xxz.shape)
    Ez_xz = field_xz_np[:, 2].reshape(Xxz.shape)
    mag_xz = np.linalg.norm(field_xz_np[:, [0, 2]], axis=1).reshape(Xxz.shape)

    # YZ slice at x0
    Yyz, Zyz = _grid(*ylim, *zlim)
    Xyz = np.full_like(Yyz, np.clip(x0, xlim[0], xlim[1]))
    coords_yz = _label_tensor_from_arrays(Xyz.ravel(), Yyz.ravel(), Zyz.ravel(), device=device, dtype=dtype)
    _, field_yz = _evaluate_field(solver, coords_yz)
    field_yz_np = field_yz.cpu().numpy()
    Ey_yz = field_yz_np[:, 1].reshape(Yyz.shape)
    Ez_yz = field_yz_np[:, 2].reshape(Yyz.shape)
    mag_yz = np.linalg.norm(field_yz_np[:, 1:], axis=1).reshape(Yyz.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    def _plot(ax, image, extent, quiv_x, quiv_y, quiv_u, quiv_v, title, xlabel, ylabel):
        im = ax.imshow(image.T, origin="lower", extent=extent, cmap="viridis")
        ax.quiver(quiv_x[::6, ::6], quiv_y[::6, ::6], quiv_u[::6, ::6], quiv_v[::6, ::6], color="w", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return im

    im0 = _plot(
        axes[0],
        mag_xy,
        (xlim[0], xlim[1], ylim[0], ylim[1]),
        Xxy,
        Yxy,
        Ex_xy,
        Ey_xy,
        f"|E| (XY @ z={np.clip(z0, zlim[0], zlim[1]):+.2f} mm)",
        "x [mm]",
        "y [mm]",
    )

    im1 = _plot(
        axes[1],
        mag_xz,
        (xlim[0], xlim[1], zlim[0], zlim[1]),
        Xxz,
        Zxz,
        Ex_xz,
        Ez_xz,
        f"|E| (XZ @ y={np.clip(y0, ylim[0], ylim[1]):+.2f} mm)",
        "x [mm]",
        "z [mm]",
    )

    im2 = _plot(
        axes[2],
        mag_yz,
        (ylim[0], ylim[1], zlim[0], zlim[1]),
        Yyz,
        Zyz,
        Ey_yz,
        Ez_yz,
        f"|E| (YZ @ x={np.clip(x0, xlim[0], xlim[1]):+.2f} mm)",
        "y [mm]",
        "z [mm]",
    )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="|E| [a.u.]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.show()


def plot_collocation_samples(
    problem: TowerALaplaceProblem,
    *,
    interior_points: int,
    contact_points: int,
    outer_points: int,
    backend: str | None = None,
) -> None:
    """Scatter representative collocation points for interior/contact/outer domains."""

    plt = _ensure_matplotlib(backend)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    def _sample(domain, n: int, mode: str) -> np.ndarray:
        sample = domain.sample(n=n, mode=mode)
        tensor = sample.tensor if isinstance(sample, LabelTensor) else sample
        return tensor.cpu().numpy()

    interior_name = problem.geometry.interior_name
    interior_domain = problem.domains[interior_name]
    P = _sample(interior_domain, interior_points, mode="latin")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.2, label="interior")

    for patch in problem.geometry.contacts:
        dom = problem.domains[patch.domain_name("contact")]
        Q = _sample(dom, contact_points, mode="random")
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=3, alpha=0.6, label=f"contact:{patch.name}")

    for patch in problem.geometry.outers:
        dom = problem.domains[patch.domain_name("outer")]
        R = _sample(dom, outer_points, mode="random")
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], s=1, alpha=0.2, label=f"outer:{patch.name}")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    ax.set_title("Representative collocation samples")
    plt.tight_layout()
    plt.show()


__all__ = ["plot_field_slices", "plot_collocation_samples"]
