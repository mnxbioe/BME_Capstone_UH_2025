"""Verification utilities for Tower A v2 (physics-relevant, fast)."""

from .physics import (
    PatchFlux,
    boundary_residual_summary,
    check_net_current_conservation,
    estimate_patch_current,
    pde_residual_summary,
)

__all__ = [
    "PatchFlux",
    "boundary_residual_summary",
    "check_net_current_conservation",
    "estimate_patch_current",
    "pde_residual_summary",
]

