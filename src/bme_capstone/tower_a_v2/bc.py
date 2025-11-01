"""Boundary-condition helpers for Tower A v2 (units explicit).

Currents are expressed in Amperes (A), areas in square millimetres (mm^2),
and flux densities in A/mm^2. Geometry is the single source of truth for area.
"""

from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Optional, Union

Number = Union[float, int]


def current_to_flux_density(current: Number, area_mm2: float) -> float:
    if area_mm2 <= 0.0:
        raise ValueError("Patch area must be positive to compute flux density.")
    return float(current) / float(area_mm2)


def canonicalise_bc_type(value: str) -> str:
    normalised = value.strip().lower()
    if normalised not in {"neumann", "dirichlet"}:
        raise ValueError(f"Unsupported boundary-condition type: {value!r}")
    return normalised


def ensure_mapping(obj: Optional[Mapping[str, Number]]) -> MutableMapping[str, Number]:
    return dict(obj) if obj is not None else {}


CallableOrNumber = Union[Number, Callable]


def s_per_m_to_s_per_mm(value_s_per_m: Number) -> float:
    """Convert a scalar conductivity from S/m to S/mm."""

    return float(value_s_per_m) / 1000.0


def um_to_mm(value_um: Number) -> float:
    """Convert micrometres to millimetres."""

    return float(value_um) / 1000.0


__all__ = [
    "CallableOrNumber",
    "Number",
    "canonicalise_bc_type",
    "current_to_flux_density",
    "ensure_mapping",
    "s_per_m_to_s_per_mm",
    "um_to_mm",
]

