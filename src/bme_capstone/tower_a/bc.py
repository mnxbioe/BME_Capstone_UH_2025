"""Boundary-condition helpers for Tower A."""

from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Optional, Union

Number = Union[float, int]


def current_to_flux_density(current: Number, area: float) -> float:
    """Convert a total electrode current into a surface flux density.

    Parameters
    ----------
    current : float | int
        Total current delivered through the patch (Amperes).
    area : float
        Surface area of the patch (square millimetres if the geometry uses
        millimetre units).

    Returns
    -------
    float
        Flux density (A/mm²).
    """
    if area <= 0.0:
        raise ValueError("Patch area must be positive to compute flux density.")
    return float(current) / float(area)


def canonicalise_bc_type(value: str) -> str:
    """Normalise a boundary-condition type string."""
    normalised = value.strip().lower()
    if normalised not in {"neumann", "dirichlet"}:
        raise ValueError(f"Unsupported boundary-condition type: {value!r}")
    return normalised


def ensure_mapping(obj: Optional[Mapping[str, Number]]) -> MutableMapping[str, Number]:
    """Return a mutable mapping, falling back to an empty dict."""
    return dict(obj) if obj is not None else {}


CallableOrNumber = Union[Number, Callable]
