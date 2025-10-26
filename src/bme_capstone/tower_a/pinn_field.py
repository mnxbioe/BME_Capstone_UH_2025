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
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.model import FeedForward
from pina.operator import div, grad
from pina.problem import SpatialProblem
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
        if tensor.shape[0] == batch and tensor.shape[1] == components:
            return tensor
        if tensor.shape[0] == 1 and tensor.shape[1] == components:
            tensor = tensor.repeat(batch, 1)
        elif tensor.shape[0] == batch and tensor.shape[1] == 1 and components > 1:
            tensor = tensor.repeat(1, components)
        else:
            raise ValueError(
                f"Expected tensor with shape (batch,{components}) or (1,{components}), got {tensor.shape}"
            )
    else:
        raise ValueError("Expected scalar, vector, or matrix representation.")

    return tensor


def _evaluate_scalar(value: BoundaryValueLike, input_: LabelTensor) -> torch.Tensor:
    """Evaluate a scalar boundary target on ``input_`` points."""
    evaluated = value(input_) if callable(value) else value
    return _as_tensor(evaluated, input_, components=1).squeeze(-1)


def _value_to_tensor(value: TensorLike, input_: LabelTensor) -> torch.Tensor:
    """Convert arbitrary value to a tensor on the same device/dtype as ``input_``."""
    if isinstance(value, LabelTensor):
        return value.tensor.to(device=input_.device, dtype=input_.dtype)
    if isinstance(value, torch.Tensor):
        return value.to(device=input_.device, dtype=input_.dtype)
    if isinstance(value, (Sequence,)):
        return torch.as_tensor(value, dtype=input_.dtype, device=input_.device)
    return torch.tensor(value, dtype=input_.dtype, device=input_.device)


def _evaluate_sigma_matrix(
    conductivity: ConductivityLike,
    input_: LabelTensor,
    axes: Axes,
) -> torch.Tensor:
    """Return an SPD conductivity tensor ``sigma`` with shape ``(batch, d, d)``."""
    value = conductivity(input_) if callable(conductivity) else conductivity
    tensor = _value_to_tensor(value, input_)
    batch = input_.shape[0]
    dim = len(axes)

    def _repeat(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape[0] == batch:
            return matrix
        if matrix.shape[0] == 1:
            return matrix.repeat(batch, 1, 1)
        raise ValueError(
            f"Conductivity batch dimension {matrix.shape[0]} incompatible with collocation batch {batch}."
        )

    if tensor.ndim == 0:
        diag = tensor.view(1).repeat(batch, dim)
        return torch.diag_embed(diag)
    if tensor.ndim == 1:
        n = tensor.shape[0]
        if n == dim:
            diag = tensor.view(1, dim).repeat(batch, 1)
            return torch.diag_embed(diag)
        if n == dim * dim:
            mats = tensor.view(1, dim, dim).repeat(batch, 1, 1)
            return mats
        if n == 1:
            diag = tensor.repeat(batch, dim)
            return torch.diag_embed(diag)
        if n == batch:
            diag = tensor.view(batch, 1).repeat(1, dim)
            return torch.diag_embed(diag)
        raise ValueError("Unsupported conductivity vector shape.")
    if tensor.ndim == 2:
        rows, cols = tensor.shape
        if rows == batch and cols == dim:
            return torch.diag_embed(tensor)
        if rows == 1 and cols == dim:
            return torch.diag_embed(tensor.repeat(batch, 1))
        if rows == dim and cols == dim:
            mats = tensor.view(1, dim, dim).repeat(batch, 1, 1)
            return mats
        if rows == batch and cols == dim * dim:
            return tensor.view(batch, dim, dim)
        if rows == 1 and cols == dim * dim:
            return tensor.view(1, dim, dim).repeat(batch, 1, 1)
        raise ValueError("Unsupported conductivity matrix shape.")
    if tensor.ndim == 3 and tensor.shape[1:] == (dim, dim):
        return _repeat(tensor)
    raise ValueError(
        "Conductivity must be scalar, diag vector, flattened matrix, or (batch,d,d) tensor."
    )


def evaluate_sigma_matrix(
    conductivity: ConductivityLike,
    input_: LabelTensor,
    axes: Axes = DEFAULT_AXES,
) -> torch.Tensor:
    """Public helper returning ``sigma`` matrices shaped ``(batch, d, d)``."""
    return _evaluate_sigma_matrix(conductivity, input_, axes)


def evaluate_sigma_diag(conductivity: ConductivityLike, input_: LabelTensor, axes: Axes) -> torch.Tensor:
    """Return only the diagonal entries of the conductivity tensor."""
    sigma = _evaluate_sigma_matrix(conductivity, input_, axes)
    diag = torch.stack([sigma[:, idx, idx] for idx in range(len(axes))], dim=1)
    return diag


def build_laplace_equation(conductivity: ConductivityLike, axes: Axes = DEFAULT_AXES) -> Equation:
    """Return the :math:`\\nabla\\cdot(\\sigma\\nabla\\phi)` residual as a PINA equation."""

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        grad_phi = grad(output_, input_, components=["phi"], d=list(axes)).tensor
        sigma_matrix = _evaluate_sigma_matrix(conductivity, input_, axes)
        flux_tensor = torch.einsum("bij,bj->bi", sigma_matrix, grad_phi)
        flux = LabelTensor(flux_tensor, labels=[f"J_{ax}" for ax in axes])
        divergence = div(flux, input_, components=flux.labels, d=list(axes))
        return divergence

    return Equation(residual)


def _compute_normal_flux(
    patch: PlanePatch,
    conductivity: ConductivityLike,
    axes: Axes,
    input_: LabelTensor,
    output_: LabelTensor,
) -> torch.Tensor:
    """Compute ``-n · σ ∇φ`` on the collocation points for a given patch."""
    grad_tensor = grad(output_, input_, components=["phi"], d=list(axes)).tensor
    sigma_matrix = _evaluate_sigma_matrix(conductivity, input_, axes)
    sigma_grad = torch.einsum("bij,bj->bi", sigma_matrix, grad_tensor)
    normal = torch.zeros(len(axes), device=input_.device, dtype=input_.dtype)
    axis_index = axes.index(patch.axis)
    normal[axis_index] = float(patch.normal_sign)
    normal_flux = -sigma_grad.matmul(normal)
    return normal_flux


def build_pointwise_neumann_equation(
    patch: PlanePatch,
    target_flux_density: BoundaryValueLike,
    conductivity: ConductivityLike,
    axes: Axes = DEFAULT_AXES,
) -> Equation:
    """Enforce ``-n·σ∇φ = target`` pointwise on ``patch``."""

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        normal_flux = _compute_normal_flux(patch, conductivity, axes, input_, output_)
        target = _evaluate_scalar(target_flux_density, input_)
        return normal_flux - target

    return Equation(residual)


def build_integral_neumann_equation(
    patch: PlanePatch,
    target_current: float,
    conductivity: ConductivityLike,
    axes: Axes = DEFAULT_AXES,
) -> Equation:
    """Enforce the integrated flux across ``patch`` equals ``target_current``."""
    area = patch.area

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        normal_flux = _compute_normal_flux(patch, conductivity, axes, input_, output_)
        integral_estimate = normal_flux.mean() * area
        return integral_estimate - float(target_current)

    return Equation(residual)


def build_dirichlet_equation(target_value: BoundaryValueLike) -> Equation:
    """Return a Dirichlet residual enforcing ``phi = target``."""

    def residual(input_: LabelTensor, output_: LabelTensor) -> LabelTensor:
        target = _evaluate_scalar(target_value, input_)
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


class TowerALaplaceProblem(SpatialProblem):
    """PINA ``SpatialProblem`` enforcing the Tower A Laplace formulation."""

    output_variables = ["phi"]

    def __init__(
        self,
        geometry: TowerAGeometry,
        conductivity: ConductivityLike,
        contact_currents: Mapping[str, float],
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
        axes: Axes = DEFAULT_AXES,
        neumann_mode: str = "pointwise",
    ) -> None:
        self._geometry = geometry
        self._axes = axes
        self._neumann_mode = self._validate_neumann_mode(neumann_mode)
        self.conductivity = conductivity
        self.contact_currents = {
            k: float(contact_currents.get(k, 0.0)) for k in (p.name for p in geometry.contacts)
        }
        self.surface_conditions: Dict[str, SurfaceConditionSummary] = {}
        self.gauge_domain_name: Optional[str] = None

        outer_specs = self._normalise_outer_bc(outer_bc or {})
        self._has_dirichlet_outer = any(spec["type"] == "dirichlet" for spec in outer_specs.values())

        if not self._has_dirichlet_outer:
            net_current = sum(self.contact_currents.values())
            if abs(net_current) > NET_CURRENT_TOL:
                raise ValueError(
                    "Net injected current must be ~0 A when no Dirichlet (ground) boundary is present."
                )

        self.domains = geometry.build_domains()
        self._spatial_domain = self.domains[geometry.interior_name]
        self._conditions = self._build_conditions(outer_specs)

        super().__init__()

    @property
    def geometry(self) -> TowerAGeometry:
        """Return the geometry used to build the problem."""
        return self._geometry

    @property
    def axes(self) -> Axes:
        """Return the spatial axes ordering."""
        return self._axes

    @property
    def spatial_domain(self):
        """Return the interior domain ``Omega``."""
        return self._spatial_domain

    @property
    def conditions(self) -> Dict[str, Condition]:
        """Return the PDE + boundary conditions."""
        return self._conditions

    @property
    def neumann_mode(self) -> str:
        """Return the Neumann enforcement mode."""
        return self._neumann_mode

    def _validate_neumann_mode(self, mode: str) -> str:
        canonical = mode.lower()
        if canonical not in {"pointwise", "integral"}:
            raise ValueError("neumann_mode must be 'pointwise' or 'integral'.")
        return canonical

    def _normalise_outer_bc(
        self,
        overrides: Mapping[str, Mapping[str, BoundaryValueLike]],
    ) -> Dict[str, Mapping[str, BoundaryValueLike]]:
        specs: Dict[str, Mapping[str, BoundaryValueLike]] = {}
        for patch in self.geometry.outers:
            spec = dict(overrides.get(patch.name, {}))
            bc_type = bc_utils.canonicalise_bc_type(spec.get("type", "dirichlet"))
            value = spec.get("value", 0.0)
            specs[patch.name] = {"type": bc_type, "value": value}
        return specs

    def _build_conditions(
        self,
        outer_specs: Mapping[str, Mapping[str, BoundaryValueLike]],
    ) -> Dict[str, Condition]:
        conditions: Dict[str, Condition] = {}
        # PDE residual
        conditions["pde"] = Condition(
            domain=self.geometry.interior_name,
            equation=build_laplace_equation(self.conductivity, self.axes),
        )

        # Contact Neumann conditions
        for patch in self.geometry.contacts:
            current = self.contact_currents.get(patch.name, 0.0)
            if self.neumann_mode == "integral":
                equation = build_integral_neumann_equation(
                    patch, current, self.conductivity, self.axes
                )
                target = current
                description = f"I = {current:.3e} A (integral)"
            else:
                flux_density = bc_utils.current_to_flux_density(current, patch.area)
                equation = build_pointwise_neumann_equation(
                    patch, flux_density, self.conductivity, self.axes
                )
                target = flux_density
                description = f"J_n = {flux_density:.3e} A/mm²"
            domain_name = patch.domain_name("contact")
            key = f"contact_flux:{patch.name}"
            conditions[key] = Condition(domain=domain_name, equation=equation)
            self.surface_conditions[key] = SurfaceConditionSummary(
                surface=patch,
                condition_type="neumann",
                target=target,
                domain_name=domain_name,
                description=description,
            )

        # Shank insulation (zero flux)
        for patch in self.geometry.shanks:
            equation = build_pointwise_neumann_equation(
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
            spec = outer_specs.get(patch.name, {"type": "dirichlet", "value": 0.0})
            bc_type = spec["type"]
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
                equation = build_pointwise_neumann_equation(
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

        if not self._has_dirichlet_outer:
            self._add_gauge_condition(conditions)

        return conditions

    def _add_gauge_condition(self, conditions: Dict[str, Condition]) -> None:
        cx, cy, cz = self.geometry.volume.center
        gauge_domain = "__gauge__"
        self.domains[gauge_domain] = CartesianDomain({"x": cx, "y": cy, "z": cz})
        self.gauge_domain_name = gauge_domain
        conditions["gauge"] = Condition(domain=gauge_domain, equation=FixedValue(0.0, components=["phi"]))


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
        neumann_mode: str = "pointwise",
    ) -> None:
        self.geometry = geometry
        self.conductivity = conductivity
        self._solver_cls = self._wrap_solver_cls(solver_cls)
        self.axes = axes
        self.neumann_mode = self._validate_neumann_mode(neumann_mode)
        self.Trainer = Trainer  # expose for legacy scripts

    @staticmethod
    def _validate_neumann_mode(mode: str) -> str:
        canonical = mode.lower()
        if canonical not in {"pointwise", "integral"}:
            raise ValueError("neumann_mode must be 'pointwise' or 'integral'.")
        return canonical

    @staticmethod
    def _wrap_solver_cls(cls: SolverClass) -> SolverClass:
        if getattr(cls, "_towera_accepts_use_lt", False):
            return cls

        class WrappedSolver(cls):  # type: ignore[misc]
            _towera_accepts_use_lt = True

            def __init__(self, *args, use_lt=None, **kwargs):
                super().__init__(*args, **kwargs)

        WrappedSolver.__name__ = cls.__name__
        WrappedSolver.__qualname__ = cls.__qualname__
        return WrappedSolver  # type: ignore[return-value]

    @property
    def solver_cls(self) -> SolverClass:
        return self._solver_cls

    # ------------------------------------------------------------------
    # Problem/model construction helpers
    # ------------------------------------------------------------------
    def build_problem(
        self,
        contact_currents: Mapping[str, float],
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
        neumann_mode: Optional[str] = None,
    ) -> TowerALaplaceProblem:
        """Instantiate a :class:`TowerALaplaceProblem` for the given pattern."""
        mode = self._validate_neumann_mode(neumann_mode) if neumann_mode else self.neumann_mode
        return TowerALaplaceProblem(
            geometry=self.geometry,
            conductivity=self.conductivity,
            contact_currents=contact_currents,
            outer_bc=outer_bc,
            axes=self.axes,
            neumann_mode=mode,
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

        contact_default = int(contact_points) if isinstance(contact_points, int) else 8_192
        shank_default = int(shank_points) if isinstance(shank_points, int) else 4_096
        outer_default = int(outer_points) if isinstance(outer_points, int) else 4_096

        for patch in problem.geometry.contacts:
            n_pts = _resolve(contact_points, patch.name, contact_default)
            _sample_domain(n_pts, boundary_mode, patch.domain_name("contact"))
        for patch in problem.geometry.shanks:
            n_pts = _resolve(shank_points, patch.name, shank_default)
            _sample_domain(n_pts, boundary_mode, patch.domain_name("shank"))
        for patch in problem.geometry.outers:
            n_pts = _resolve(outer_points, patch.name, outer_default)
            _sample_domain(n_pts, boundary_mode, patch.domain_name("outer"))

        gauge_domain = getattr(problem, "gauge_domain_name", None)
        if gauge_domain:
            problem.discretise_domain(n=1, mode=boundary_mode, domains=gauge_domain)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_basis(
        self,
        contact_name: str,
        current: float = 1e-6,
        *,
        outer_bc: Optional[Mapping[str, Mapping[str, BoundaryValueLike]]] = None,
        neumann_mode: Optional[str] = None,
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

        problem = self.build_problem(contact_currents, outer_bc=outer_bc, neumann_mode=neumann_mode)

        discretisation_kwargs = discretisation_kwargs or {}
        self.discretise(problem, **discretisation_kwargs)

        model = model or self.build_model()

        solver_kwargs = solver_kwargs or {}
        solver = self.solver_cls(problem=problem, model=model, **solver_kwargs)

        trainer_kwargs = {
            "max_epochs": 2_000,
            "enable_model_summary": False,
        } | (trainer_kwargs or {})
        trainer_kwargs.setdefault("accelerator", "auto")
        trainer_kwargs.setdefault("devices", 1)
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
