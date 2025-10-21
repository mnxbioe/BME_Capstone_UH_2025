"""PINN scaffolding for Tower A's physics-informed field encoder.

This module translates the Tower A methodology into concrete PINA components:

* :class:`TowerALaplaceProblem` configures the Laplace PDE with mixed boundary
  conditions given a :class:`~bme_capstone.tower_a.geometry.TowerAGeometry`
  instance.
* :class:`TowerABasisTrainer` provides a convenience wrapper to train unit
  current basis fields (``phi_k``) using PINA solvers, handling common defaults
  such as discretisation, model construction, and training loops.

The code intentionally exposes low-level handles (PINA ``Problem``/``Solver``/
``Trainer`` objects) so that advanced users can swap optimisers, weighting
schemes (e.g. ``SelfAdaptivePINN``), or callbacks while preserving the physics
formulation described in the manuscript's Methods section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn

from pina import Condition, LabelTensor
from pina.equation import Equation, FixedValue
from pina.model import FeedForward
from pina.operator import div, grad
from pina.solver.physics_informed_solver import PINN
from pina.solver.physics_informed_solver.rba_pinn import RBAPINN
from pina.solver.physics_informed_solver.self_adaptive_pinn import SelfAdaptivePINN
from pina.trainer import Trainer

from . import bc as bc_utils
from .geometry import PlanePatch, TowerAGeometry

Axes = Tuple[str, ...]
TensorLike = Union[float, int, Sequence[float], torch.Tensor, LabelTensor]
ConductivityLike = Union[TensorLike, Callable[[LabelTensor], TensorLike]]
BoundaryValueLike = Union[TensorLike, Callable[[LabelTensor], TensorLike]]

SolverType = Union[PINN, SelfAdaptivePINN, RBAPINN]
SolverClass = Union[type[PINN], type[SelfAdaptivePINN], type[RBAPINN]]

DEFAULT_AXES: Axes = ("x", "y", "z")


def _as_tensor(value: TensorLike, input_: LabelTensor, components: int = 1) -> torch.Tensor:
    """Convert ``value`` into a tensor aligned with ``input_``'s batch."""
    device = input_.device
    dtype = input_.dtype
    batch = input_.shape[0]

    if isinstance(value, LabelTensor):
        tensor = value.tensor.to(device=device, dtype=dtype)
    elif isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    elif isinstance(value, (Sequence,)):
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
    else:
        tensor = torch.tensor(value, dtype=dtype, device=device)

    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).repeat(batch, components)
    elif tensor.ndim == 1:
        if tensor.shape[0] == batch and components == 1:
            tensor = tensor.reshape(batch, 1)
        elif tensor.shape[0] == components:
            tensor = tensor.reshape(1, components).repeat(batch, 1)
        elif tensor.shape[0] == batch and components != 1:
            tensor = tensor.reshape(batch, 1).repeat(1, components)
        else:
            raise ValueError(
                f"Cannot reshape tensor of shape {tuple(tensor.shape)} to ({batch}, {components})"
            )
    elif tensor.ndim == 2:
        if tensor.shape[0] != batch:
            raise ValueError(
                f"Expected first dimension equal to batch size {batch}, got {tensor.shape}"
            )
        if tensor.shape[1] != components:
            if tensor.shape[1] == 1 and components > 1:
                tensor = tensor.repeat(1, components)
            else:
                raise ValueError(
                    f"Expected second dimension {components}, got {tensor.shape}"
                )
    else:
        raise ValueError("Expected scalar, vector, or matrix conductivity representation.")

    return tensor


def evaluate_sigma_diag(conductivity: ConductivityLike, input_: LabelTensor, axes: Axes) -> torch.Tensor:
    """Evaluate (potentially spatially varying) conductivity on ``input_`` points.

    Returns an ``(batch, len(axes))`` tensor corresponding to the diagonal of
    :math:`\\boldsymbol{\\sigma}` in the coordinate system spanned by ``axes``.
    """
    if callable(conductivity):
        value = conductivity(input_)
    else:
        value = conductivity
    sigma = _as_tensor(value, input_, components=len(axes))
    if sigma.shape[1] == 1 and len(axes) > 1:
        sigma = sigma.repeat(1, len(axes))
    return sigma


def evaluate_scalar(value: BoundaryValueLike, input_: LabelTensor) -> torch.Tensor:
    """Evaluate a scalar boundary target on ``input_`` points."""
    if callable(value):
        evaluated = value(input_)
    else:
        evaluated = value
    return _as_tensor(evaluated, input_, components=1).squeeze(-1)


def build_laplace_equation(conductivity: ConductivityLike, axes: Axes = DEFAULT_AXES) -> Equation:
    """Return the :math:`\\nabla\\cdot(\\sigma\\nabla\\phi)` residual as a PINA equation."""

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        grad_phi = grad(output_, input_, components=["phi"], d=list(axes))
        sigma_diag = evaluate_sigma_diag(conductivity, input_, axes)
        flux_tensor = sigma_diag * grad_phi.tensor
        flux = LabelTensor(
            flux_tensor, labels=[f"J_{ax}" for ax in axes]
        )
        divergence = div(flux, input_, components=flux.labels, d=list(axes))
        return divergence

    return Equation(residual)


def build_neumann_equation(
    patch: PlanePatch,
    target_flux_density: BoundaryValueLike,
    conductivity: ConductivityLike,
    axes: Axes = DEFAULT_AXES,
) -> Equation:
    """Return the Neumann residual enforcing ``-n·σ∇φ = target`` on a patch."""
    axis = patch.axis
    if axis not in axes:
        raise ValueError(f"Patch axis {axis!r} not present in axes {axes}.")
    axis_index = axes.index(axis)

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        grad_component = grad(output_, input_, components=["phi"], d=[axis])
        sigma_diag = evaluate_sigma_diag(conductivity, input_, axes)
        sigma_axis = sigma_diag[:, axis_index]
        normal_flux = -patch.normal_sign * sigma_axis * grad_component.tensor.squeeze(-1)
        target = evaluate_scalar(target_flux_density, input_)
        return normal_flux - target

    return Equation(residual)


def build_dirichlet_equation(target_value: BoundaryValueLike) -> Equation:
    """Return a Dirichlet residual enforcing ``phi = target``."""

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        target = evaluate_scalar(target_value, input_)
        phi = output_.extract(["phi"]).tensor.squeeze(-1)
        return phi - target

    return Equation(residual)


@dataclass
class SurfaceConditionSummary:
    """Metadata describing the condition attached to a boundary surface."""

    surface: PlanePatch
    condition_type: str
    target: BoundaryValueLike
    domain_name: str
    description: str


class TowerALaplaceProblem(PINN.problem.__class__ if False else object):
    """PINA ``SpatialProblem`` enforcing the Tower A Laplace formulation."""

    output_variables = ["phi"]

    def __init__(
        self,
        geometry: TowerAGeometry,
        conductivity: ConductivityLike,
        contact_currents: Mapping[str, float],
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
        axes: Axes = DEFAULT_AXES,
    ) -> None:
        from pina.problem import SpatialProblem  # local import to avoid circularity

        class _Problem(SpatialProblem):  # type: ignore[valid-type]
            pass

        # -------------------------------
        # 1. Store geometry and materials
        # -------------------------------
        self._geometry = geometry
        self._axes = axes
        self.conductivity = conductivity

        # -------------------------------
        # 2. Define contact currents FIRST
        # -------------------------------
        self.contact_currents = {
            k: float(contact_currents.get(k, 0.0)) for k in (p.name for p in geometry.contacts)
        }

        # ✅ define these BEFORE building conditions
        self.contact_flux_densities: Dict[str, float] = {}
        self.surface_conditions: Dict[str, SurfaceConditionSummary] = {}

        # -------------------------------
        # 3. Build geometry domains
        # -------------------------------
        self.domains = geometry.build_domains()
        self.spatial_domain = self.domains[geometry.interior_name]

        # -------------------------------
        # 4. Build PDE + boundary conditions
        # -------------------------------
        self._conditions = self._build_conditions(outer_bc or {})

        # -------------------------------
        # 5. Initialize PINA base problem
        # -------------------------------
        super().__init__()

    @property
    def conditions(self):
        """Return the boundary and PDE conditions dictionary."""
        return self._conditions

    @property
    def geometry(self) -> TowerAGeometry:
        """Return the geometry used to build the problem."""
        return self._geometry

    @property
    def axes(self) -> Axes:
        """Return the spatial axes ordering."""
        return self._axes

    def _build_conditions(
        self,
        outer_bc: Mapping[str, Mapping[str, BoundaryValueLike]],
    ) -> Dict[str, Condition]:
        conditions: Dict[str, Condition] = {}
        # PDE residual
        conditions["pde"] = Condition(
            domain=self.geometry.interior_name,
            equation=build_laplace_equation(self.conductivity, self.axes),
        )

        # Contact Neumann conditions
        for patch in self.geometry.contacts:
            flux_density = bc_utils.current_to_flux_density(
                self.contact_currents.get(patch.name, 0.0), patch.area
            )
            equation = build_neumann_equation(
                patch, flux_density, self.conductivity, self.axes
            )
            domain_name = patch.domain_name("contact")
            key = f"contact_flux:{patch.name}"
            conditions[key] = Condition(domain=domain_name, equation=equation)
            self.surface_conditions[key] = SurfaceConditionSummary(
                surface=patch,
                condition_type="neumann",
                target=flux_density,
                domain_name=domain_name,
                description=f"I={self.contact_currents.get(patch.name, 0.0)} A",
            )

        # Shank insulation (zero flux)
        for patch in self.geometry.shanks:
            equation = build_neumann_equation(
                patch, 0.0, self.conductivity, self.axes
            )
            domain_name = patch.domain_name("shank")
            key = f"shank_flux:{patch.name}"
            conditions[key] = Condition(domain=domain_name, equation=equation)
            self.surface_conditions[key] = SurfaceConditionSummary(
                surface=patch,
                condition_type="neumann",
                target=0.0,
                domain_name=domain_name,
                description="shank insulation",
            )

        # Outer boundary (Dirichlet or optional Neumann)
        for patch in self.geometry.outers:
            spec = outer_bc.get(patch.name, {"type": "dirichlet", "value": 0.0})
            bc_type = bc_utils.canonicalise_bc_type(spec.get("type", "dirichlet"))
            domain_name = patch.domain_name("outer")
            key = f"outer_bc:{patch.name}"

            if bc_type == "dirichlet":
                value = spec.get("value", 0.0)
                if isinstance(value, (int, float)):
                    equation = FixedValue(float(value), components=["phi"])
                else:
                    equation = build_dirichlet_equation(value)
                description = f"Dirichlet φ={value}"
            else:
                value = spec.get("value", 0.0)
                equation = build_neumann_equation(
                    patch, value, self.conductivity, self.axes
                )
                description = f"Neumann flux={value}"

            conditions[key] = Condition(domain=domain_name, equation=equation)
            self.surface_conditions[key] = SurfaceConditionSummary(
                surface=patch,
                condition_type=bc_type,
                target=value,
                domain_name=domain_name,
                description=description,
            )

        return conditions


@dataclass
class BasisTrainingResult:
    """Container for artefacts generated while training a basis field."""

    contact_name: str
    current: float
    problem: TowerALaplaceProblem
    model: nn.Module
    solver: SolverType
    trainer: Trainer


class TowerABasisTrainer:
    """Utility to train unit-current basis fields with sensible defaults."""

    def __init__(
        self,
        geometry: TowerAGeometry,
        conductivity: ConductivityLike,
        solver_cls: SolverClass = PINN,
        axes: Axes = DEFAULT_AXES,
    ) -> None:
        self.geometry = geometry
        self.conductivity = conductivity
        self.solver_cls = solver_cls
        self.axes = axes

    # ------------------------------------------------------------------
    # Problem/model construction helpers
    # ------------------------------------------------------------------
    def build_problem(
        self,
        contact_currents: Mapping[str, float],
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
    ) -> TowerALaplaceProblem:
        """Instantiate a :class:`TowerALaplaceProblem` for the given pattern."""
        return TowerALaplaceProblem(
            geometry=self.geometry,
            conductivity=self.conductivity,
            contact_currents=contact_currents,
            outer_bc=outer_bc,
            axes=self.axes,
        )

    def build_model(
        self,
        n_layers: int = 6,
        hidden_size: int = 128,
        activation: type[nn.Module] = nn.Tanh,
        fourier_features: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Return the default smooth MLP used for basis fields."""
        model = FeedForward(
            input_dimensions=len(self.axes),
            output_dimensions=1,
            inner_size=hidden_size,
            n_layers=n_layers,
            func=activation,
        )
        if fourier_features is not None:
            # Simple wrapper: pass inputs through features before MLP
            return nn.Sequential(fourier_features, model)
        return model

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------
    def discretise(
        self,
        problem: TowerALaplaceProblem,
        interior_points: int = 50_000,
        contact_points: Union[int, Mapping[str, int]] = 8_192,
        shank_points: Union[int, Mapping[str, int]] = 4_096,
        outer_points: Union[int, Mapping[str, int]] = 4_096,
        interior_mode: str = "latin",
        boundary_mode: str = "random",
    ) -> None:
        """Sample all domains with reasonable defaults."""

        def _sample_domain(n: int, mode: str, domain: str) -> None:
            if n <= 0:
                n = 1  # ensure at least one point per domain
            problem.discretise_domain(n=n, mode=mode, domains=domain)

        _sample_domain(interior_points, interior_mode, problem.geometry.interior_name)

        def _resolve(mapping_or_int: Union[int, Mapping[str, int]], name: str, default: int) -> int:
            if isinstance(mapping_or_int, Mapping):
                return int(mapping_or_int.get(name, default))
            return int(mapping_or_int)

        for patch in problem.geometry.contacts:
            n_pts = _resolve(contact_points, patch.name, int(contact_points))
            _sample_domain(n_pts, boundary_mode, patch.domain_name("contact"))
        for patch in problem.geometry.shanks:
            n_pts = _resolve(shank_points, patch.name, int(shank_points))
            _sample_domain(n_pts, boundary_mode, patch.domain_name("shank"))
        for patch in problem.geometry.outers:
            n_pts = _resolve(outer_points, patch.name, int(outer_points))
            _sample_domain(n_pts, boundary_mode, patch.domain_name("outer"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_basis(
        self,
        contact_name: str,
        current: float = 1e-6,
        *,
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
        solver_kwargs: Optional[Dict] = None,
        trainer_kwargs: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        discretisation_kwargs: Optional[Dict] = None,
    ) -> BasisTrainingResult:
        """Train a unit-current basis field for the specified contact."""
        if contact_name not in {patch.name for patch in self.geometry.contacts}:
            raise ValueError(f"Unknown contact name {contact_name!r}.")

        contact_currents = {patch.name: 0.0 for patch in self.geometry.contacts}
        contact_currents[contact_name] = current

        problem = self.build_problem(contact_currents, outer_bc=outer_bc)

        discretisation_kwargs = discretisation_kwargs or {}
        self.discretise(problem, **discretisation_kwargs)

        model = model or self.build_model()

        solver_kwargs = solver_kwargs or {}
        solver = self.solver_cls(problem=problem, model=model, **solver_kwargs)

        trainer_kwargs = {
            "max_epochs": 2_000,
            "enable_model_summary": False,
            "accelerator": "cpu",
        } | (trainer_kwargs or {})
        trainer = Trainer(
            solver=solver,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            batch_size=None,
            **trainer_kwargs,
        )

        trainer.train()

        return BasisTrainingResult(
            contact_name=contact_name,
            current=current,
            problem=problem,
            model=model,
            solver=solver,
            trainer=trainer,
        )
if __name__ == "__main__":
    print("Tower-A PINN field module loaded successfully 🚀")
