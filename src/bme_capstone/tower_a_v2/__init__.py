"""Tower A v2 package (physics-first).

Geometry, boundary helpers, PINN scaffolding, features, verification, and
run logging live here. Import callers should use this namespace.
"""

from .geometry import Box3D, PlanePatch, TowerAGeometry

__all__ = [
    "Box3D",
    "PlanePatch",
    "TowerAGeometry",
]

