"""Physics checks for Tower A v2 (fast and relevant).

Implements:
- Net-contact current conservation via Monte Carlo integration of normal flux
- Interior PDE residual summary
- Boundary flux residual summary per contact
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from pina import LabelTensor

from bme_capstone.tower_a_v2.geometry import Box3D, PlanePatch, TowerAGeometry
from bme_capstone.tower_a_v2.pinn_field import DEFAULT_AXES, SolverType, evaluate_sigma_matrix


@dataclass
class PatchFlux:
    name: str
    area_mm2: float
    integrated_current_A: float
    mean_flux_A_per_mm2: float


def _sample_patch_points(patch: PlanePatch, n: int, axes: Tuple[str, ...]) -> LabelTensor:
    rng = np.random.default_rng(42)
    other_axes = [ax for ax in axes if ax != patch.axis]
    values = {}
    for ax in other_axes:
        lo, hi = patch.span[ax]
        values[ax] = rng.uniform(lo, hi, size=n)
    fixed_axis = np.full(n, patch.value)
    coords = []
    for ax in axes:
        if ax == patch.axis:
            coords.append(fixed_axis)
        else:
            coords.append(values[ax])
    arr = np.stack(coords, axis=1).astype(np.float32)
    tensor = torch.from_numpy(arr)
    tensor.requires_grad_(True)
    return LabelTensor(tensor, list(axes))


def _compute_normal_flux(patch: PlanePatch, conductivity, axes: Tuple[str, ...], inputs: LabelTensor, phi: LabelTensor) -> torch.Tensor:
    from pina.operator import grad

    grad_tensor = grad(phi, inputs, components=["phi"], d=list(axes)).tensor
    sigma_matrix = evaluate_sigma_matrix(conductivity, inputs, axes)
    sigma_grad = torch.einsum("bij,bj->bi", sigma_matrix, grad_tensor)
    normal = torch.zeros(len(axes), device=inputs.device, dtype=inputs.dtype)
    normal[axes.index(patch.axis)] = float(patch.normal_sign)
    return -sigma_grad.matmul(normal)


def estimate_patch_current(
    solver: SolverType,
    patch: PlanePatch,
    conductivity,
    *,
    axes: Tuple[str, ...] = DEFAULT_AXES,
    n_points: int = 4096,
) -> PatchFlux:
    solver.eval()
    inputs = _sample_patch_points(patch, n_points, axes)
    phi = solver.forward(inputs).extract(["phi"])
    normal_flux = _compute_normal_flux(patch, conductivity, axes, inputs, phi)
    mean_flux = normal_flux.mean()
    integrated = float(mean_flux) * patch.area
    return PatchFlux(
        name=patch.name,
        area_mm2=patch.area,
        integrated_current_A=float(integrated),
        mean_flux_A_per_mm2=float(mean_flux),
    )


def check_net_current_conservation(
    geometry: TowerAGeometry,
    solvers: Mapping[str, SolverType],
    conductivity,
    *,
    axes: Tuple[str, ...] = DEFAULT_AXES,
    n_points: int = 4096,
) -> Dict:
    rows = []
    for patch in geometry.contacts:
        solver = solvers.get(patch.name)
        if solver is None:
            continue
        rows.append(estimate_patch_current(solver, patch, conductivity, axes=axes, n_points=n_points))

    total = sum(r.integrated_current_A for r in rows)
    table = [
        {
            "contact": r.name,
            "area_mm2": r.area_mm2,
            "integrated_A": r.integrated_current_A,
            "mean_flux_A_per_mm2": r.mean_flux_A_per_mm2,
        }
        for r in rows
    ]
    return {"per_contact": table, "net_contact_current_A": total}


def _sample_interior_points(box: Box3D, n: int, axes: Tuple[str, ...]) -> LabelTensor:
    rng = np.random.default_rng(123)
    coords = []
    for ax in axes:
        lo, hi = getattr(box, ax)
        coords.append(rng.uniform(lo, hi, size=n))
    arr = np.stack(coords, axis=1).astype(np.float32)
    tensor = torch.from_numpy(arr)
    tensor.requires_grad_(True)
    return LabelTensor(tensor, list(axes))


def pde_residual_summary(
    geometry: TowerAGeometry,
    solver: SolverType,
    conductivity,
    *,
    axes: Tuple[str, ...] = DEFAULT_AXES,
    n_points: int = 8192,
) -> Dict:
    from pina.operator import div, grad

    solver.eval()
    inputs = _sample_interior_points(geometry.volume, n_points, axes)
    phi = solver.forward(inputs).extract(["phi"])
    grad_phi = grad(phi, inputs, components=["phi"], d=list(axes)).tensor
    sigma_matrix = evaluate_sigma_matrix(conductivity, inputs, axes)
    flux_tensor = torch.einsum("bij,bj->bi", sigma_matrix, grad_phi)
    flux = LabelTensor(flux_tensor, labels=[f"J_{ax}" for ax in axes])
    div_flux = div(flux, inputs, components=flux.labels, d=list(axes)).tensor.squeeze(-1)
    values = np.abs(div_flux.detach().cpu().numpy())
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return {"median": median, "iqr": [q1, q3], "n": int(n_points)}


def boundary_residual_summary(
    geometry: TowerAGeometry,
    solver: SolverType,
    conductivity,
    contact_currents: Mapping[str, float],
    *,
    axes: Tuple[str, ...] = DEFAULT_AXES,
    n_points: int = 4096,
) -> Dict:
    rows = []
    for patch in geometry.contacts:
        solver.eval()
        inputs = _sample_patch_points(patch, n_points, axes)
        phi = solver.forward(inputs).extract(["phi"])
        normal_flux = _compute_normal_flux(patch, conductivity, axes, inputs, phi)
        target = float(contact_currents.get(patch.name, 0.0)) / float(patch.area)
        residual = normal_flux - target
        values = np.abs(residual.detach().cpu().numpy())
        median = float(np.median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        rows.append(
            {
                "contact": patch.name,
                "median": median,
                "iqr": [q1, q3],
                "n": int(n_points),
            }
        )
    return {"contacts": rows}


__all__ = [
    "PatchFlux",
    "boundary_residual_summary",
    "check_net_current_conservation",
    "estimate_patch_current",
    "pde_residual_summary",
]

