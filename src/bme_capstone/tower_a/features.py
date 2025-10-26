"""Evaluation and feature-extraction utilities for Tower A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from pina import LabelTensor
from pina.operator import grad

from .geometry import Box3D, TowerAGeometry
from .pinn_field import (
    Axes,
    BasisTrainingResult,
    ConductivityLike,
    DEFAULT_AXES,
    SolverType,
    evaluate_sigma_matrix,
)

Number = Union[int, float]


def generate_cartesian_grid(
    box: Box3D,
    spacing: float,
    padding: float = 0.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Generate a Cartesian mesh covering ``box`` with optional padding."""
    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    x_min, x_max = box.x #Box stores its bounds as tuples (lo, hi).
    y_min, y_max = box.y
    z_min, z_max = box.z

    pad = float(padding) #The + spacing / 2 ensures the upper bound is included vvvv
    xs = torch.arange(x_min - pad, x_max + pad + spacing / 2, spacing, dtype=dtype, device=device)
    ys = torch.arange(y_min - pad, y_max + pad + spacing / 2, spacing, dtype=dtype, device=device)
    zs = torch.arange(z_min - pad, z_max + pad + spacing / 2, spacing, dtype=dtype, device=device)

    # Build the 3D mesh :
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack( # .reshape(-1) flattens each 3D array into a 1-D vector of all points.
        (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)),
        dim=-1, 
    ) 
    shape = (len(xs), len(ys), len(zs)) #combine them into an (N, 3) tensor where each row = (x, y, z) 
    return coords, shape


@dataclass
class FieldEvaluation:
    """container for evaluated field quantities."""

    coords: LabelTensor
    currents: torch.Tensor
    potential: LabelTensor
    electric_field: Optional[LabelTensor] = None
    electric_magnitude: Optional[LabelTensor] = None
    current_density: Optional[LabelTensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return tensors suitable for caching"""
        data: Dict[str, torch.Tensor] = {
            "coords": self.coords.tensor,
            "currents": self.currents,
            "phi": self.potential.tensor,
        }
        if self.electric_field is not None:
            data["E"] = self.electric_field.tensor
        if self.electric_magnitude is not None:
            data["|E|"] = self.electric_magnitude.tensor
        if self.current_density is not None:
            data["J"] = self.current_density.tensor
        return {k: v.detach().clone() for k, v in data.items()}


class FieldEvaluator:
    """Superpose trained basis fields and extract physics-aware features."""

    def __init__(
        self,
        geometry: TowerAGeometry,
        #basis is a dictionary mapping each contact name (string) to either a trained solver object or a full training result bundle.”
        basis: Mapping[str, Union[SolverType, BasisTrainingResult]], #
        conductivity: Optional[ConductivityLike],
        *,
        axes: Axes = DEFAULT_AXES,
        corrector: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.geometry = geometry
        self.axes = axes
        self.conductivity = conductivity
        self.corrector = corrector
        self.device = device
        self.dtype = dtype

        contact_order = [patch.name for patch in geometry.contacts]
        missing = set(contact_order) - set(basis.keys())
        if missing:
            raise ValueError(f"Basis dictionary missing contacts: {sorted(missing)}")

        def _unwrap(entry: Union[SolverType, BasisTrainingResult]) -> SolverType:
            if isinstance(entry, BasisTrainingResult):
                return entry.solver
            return entry

        self.contact_order = contact_order
        self.basis = {name: _unwrap(basis[name]) for name in contact_order}

    @property
    def n_contacts(self) -> int:
        """Return the number of basis fields handled by the evaluator."""
        return len(self.contact_order)

    def _standardise_currents( # ensures current tensor has right shape and matches # of contacts
        self,
        currents: Union[Sequence[Number], torch.Tensor],
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(currents, dtype=dtype, device=device)
        if tensor.ndim == 1: # for 1D cases 
            if tensor.shape[0] != self.n_contacts:
                raise ValueError(
                    f"Expected {self.n_contacts} currents, got {tensor.shape[0]}"
                )
            tensor = tensor.unsqueeze(0).repeat(n_points, 1)
        elif tensor.ndim == 2: # for 2D cases 
            if tensor.shape[1] != self.n_contacts:
                raise ValueError(
                    f"Expected current vectors of length {self.n_contacts}, got {tensor.shape[1]}"
                )
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(n_points, 1)
            elif tensor.shape[0] != n_points:
                raise ValueError(
                    f"Current batch size {tensor.shape[0]} mismatches coordinate batch {n_points}"
                )
        else:
            raise ValueError("Currents must be 1-D or 2-D tensor-like.")
        return tensor
    #used to evaluate network at specified points in space ( inputs coordiantes and currents)
    def evaluate( 
        self,
        coords: Union[Sequence[Sequence[Number]], torch.Tensor],
        currents: Union[Sequence[Number], torch.Tensor],
        *,
        compute_gradients: bool = True,
        detach: bool = True,
    ) -> FieldEvaluation:
        """Evaluate ``phi`` (and ``E``/``J``) at arbitrary points."""
        device = self.device or (coords.device if isinstance(coords, torch.Tensor) else torch.device("cpu"))
        dtype = self.dtype if not isinstance(coords, torch.Tensor) else coords.dtype

        coords_tensor = torch.as_tensor(coords, dtype=dtype, device=device)
        coords_tensor = coords_tensor.reshape(-1, len(self.axes))
        coords_tensor.requires_grad_(compute_gradients) #whether to track derivatives of φ with respect to x,y,z.
        #For PINA: Wrap the raw tensor in a LabelTensor, to track variable ("x", "y", "z").
        coords_lt = LabelTensor(coords_tensor, list(self.axes))
        #properly shape tesnor compatible with coord
        currents_tensor = self._standardise_currents(currents, coords_tensor.shape[0], device, dtype)

        phi_total = torch.zeros(
            coords_tensor.shape[0], 1, device=device, dtype=dtype
        )

        for idx, name in enumerate(self.contact_order):
            solver = self.basis[name]
            solver.eval()
            phi_k = solver.forward(coords_lt).extract(["phi"]).tensor
            phi_total = phi_total + phi_k * currents_tensor[:, idx].unsqueeze(-1)

        if self.corrector is not None:
            correct_input = torch.cat([coords_tensor, currents_tensor], dim=-1)
            phi_total = phi_total + self.corrector(correct_input)

        phi = LabelTensor(phi_total, ["phi"])

        e_field: Optional[LabelTensor] = None
        e_mag: Optional[LabelTensor] = None
        current_density: Optional[LabelTensor] = None

        if compute_gradients:
            grad_phi = grad(phi, coords_lt, components=["phi"], d=list(self.axes))
            e_tensor = -grad_phi.tensor
            e_field = LabelTensor(e_tensor, [f"E_{ax}" for ax in self.axes])

            magnitude = torch.linalg.norm(e_tensor, dim=-1, keepdim=True)
            e_mag = LabelTensor(magnitude, ["|E|"])

            if self.conductivity is not None:
                sigma_matrix = evaluate_sigma_matrix(self.conductivity, coords_lt, self.axes)
                j_tensor = torch.einsum("bij,bj->bi", sigma_matrix, e_tensor)
                current_density = LabelTensor(
                    j_tensor,
                    [f"J_{ax}" for ax in self.axes],
                )

        if detach:
            phi = phi.detach()
            coords_lt = coords_lt.detach()
            currents_tensor = currents_tensor.detach()
            if e_field is not None:
                e_field = e_field.detach()
            if e_mag is not None:
                e_mag = e_mag.detach()
            if current_density is not None:
                current_density = current_density.detach()

        return FieldEvaluation(
            coords=coords_lt,
            currents=currents_tensor,
            potential=phi,
            electric_field=e_field,
            electric_magnitude=e_mag,
            current_density=current_density,
        )

    def evaluate_on_grid(
        self,
        spacing: float,
        currents: Union[Sequence[Number], torch.Tensor],
        *,
        padding: float = 0.0,
        compute_gradients: bool = True,
        detach: bool = True,
    ) -> Tuple[FieldEvaluation, Tuple[int, int, int]]:
        """Evaluate on a uniform grid centred on the tissue volume."""
        coords, shape = generate_cartesian_grid(
            self.geometry.volume,
            spacing=spacing,
            padding=padding,
            dtype=self.dtype,
            device=self.device,
        )
        evaluation = self.evaluate(
            coords, currents, compute_gradients=compute_gradients, detach=detach
        )
        return evaluation, shape
