"""Tower A physics-informed field encoder utilities."""

from .geometry import Box3D, PlanePatch, TowerAGeometry,outer_faces_from_box,make_gauge_patch  
from .bc import canonicalise_bc_type, current_to_flux_density, ensure_mapping
from .pinn_field import (
    DEFAULT_AXES,
    Axes,
    BasisTrainingResult,
    ConductivityLike,
    TowerABasisTrainer,
    TowerALaplaceProblem,
    evaluate_sigma_diag,
    evaluate_sigma_matrix,
)
from .features import FieldEvaluation, FieldEvaluator, generate_cartesian_grid

__all__ = [
    "Axes",
    "Box3D",
    "PlanePatch",
    "TowerAGeometry",
    "outer_faces_from_box",
    "make_gauge_patch",
    "TowerALaplaceProblem",
    "TowerABasisTrainer",
    "BasisTrainingResult",
    "ConductivityLike",
    "DEFAULT_AXES",
    "evaluate_sigma_diag",
    "evaluate_sigma_matrix",
    "FieldEvaluator",
    "FieldEvaluation",
    "generate_cartesian_grid",
    "current_to_flux_density",
    "canonicalise_bc_type",
    "ensure_mapping",
]
