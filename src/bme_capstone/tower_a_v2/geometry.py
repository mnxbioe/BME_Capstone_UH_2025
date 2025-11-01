"""Tower A v2 geometry helpers (geometry is code).

All lengths are expressed in millimetres. Plane patches track outward normals
and areas so boundary flux calculations remain explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from pina.domain import CartesianDomain


Axis = str  # restricted to {"x", "y", "z"}
_AXES: Tuple[Axis, Axis, Axis] = ("x", "y", "z")


def _validate_axis(axis: Axis) -> Axis:
    if axis not in _AXES:
        raise ValueError(f"Expected axis in {_AXES}, got {axis!r}")
    return axis


def _ensure_limits(bounds: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError(f"Bounds must satisfy lo < hi, got ({lo}, {hi})")
    return lo, hi


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned box describing the tissue volume (mm)."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _ensure_limits(self.x))
        object.__setattr__(self, "y", _ensure_limits(self.y))
        object.__setattr__(self, "z", _ensure_limits(self.z))

    def to_domain(self) -> CartesianDomain:
        return CartesianDomain({"x": list(self.x), "y": list(self.y), "z": list(self.z)})

    @property
    def extents(self) -> Tuple[float, float, float]:
        return (
            self.x[1] - self.x[0],
            self.y[1] - self.y[0],
            self.z[1] - self.z[0],
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x[0] + self.x[1]) * 0.5,
            (self.y[0] + self.y[1]) * 0.5,
            (self.z[0] + self.z[1]) * 0.5,
        )


@dataclass
class PlanePatch:
    """Axis-aligned planar patch representing a contact, shank, or boundary."""

    name: str
    axis: Axis
    value: float
    span: Mapping[Axis, Tuple[float, float]]
    normal_sign: int = 1
    kind: str = "contact"
    metadata: MutableMapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.axis = _validate_axis(self.axis)
        if self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be -1 or +1")
        expected_other_axes = {ax for ax in _AXES if ax != self.axis}
        if set(self.span.keys()) != expected_other_axes:
            raise ValueError(
                f"span must provide bounds for axes {expected_other_axes}, got keys={set(self.span.keys())}"
            )
        cleaned: Dict[Axis, Tuple[float, float]] = {}
        for ax, bounds in self.span.items():
            cleaned[ax] = _ensure_limits(bounds)
        object.__setattr__(self, "span", cleaned)

    @property
    def area(self) -> float:
        lengths = [bounds[1] - bounds[0] for bounds in self.span.values()]
        return lengths[0] * lengths[1]

    @property
    def normal(self) -> Tuple[float, float, float]:
        vec = [0.0, 0.0, 0.0]
        vec[_AXES.index(self.axis)] = float(self.normal_sign)
        return tuple(vec)  # type: ignore[return-value]

    def to_domain(self) -> CartesianDomain:
        payload = {self.axis: float(self.value)}
        for ax, bounds in self.span.items():
            payload[ax] = [float(bounds[0]), float(bounds[1])]
        return CartesianDomain(payload)

    def domain_name(self, prefix: Optional[str] = None) -> str:
        tag = prefix or self.kind
        return f"{tag}:{self.name}"


@dataclass
class TowerAGeometry:
    """Group the interior volume and planar surface patches."""

    volume: Box3D
    contacts: List[PlanePatch]
    shanks: List[PlanePatch] = field(default_factory=list)
    outers: List[PlanePatch] = field(default_factory=list)
    interior_name: str = "interior"
    gauge: Optional[PlanePatch] = None

    def build_domains(self) -> Dict[str, CartesianDomain]:
        domains: Dict[str, CartesianDomain] = {self.interior_name: self.volume.to_domain()}
        for patch in self.contacts:
            domains[patch.domain_name("contact")] = patch.to_domain()
        for patch in self.shanks:
            domains[patch.domain_name("shank")] = patch.to_domain()
        for patch in self.outers:
            domains[patch.domain_name("outer")] = patch.to_domain()
        if self.gauge is not None:
            domains[self.gauge.domain_name("gauge")] = self.gauge.to_domain()
        return domains

    def iter_contacts(self) -> Iterable[PlanePatch]:
        return iter(self.contacts)

    def iter_shanks(self) -> Iterable[PlanePatch]:
        return iter(self.shanks)

    def iter_outers(self) -> Iterable[PlanePatch]:
        return iter(self.outers)

    def all_surfaces(self) -> Iterable[PlanePatch]:
        yield from self.contacts
        yield from self.shanks
        yield from self.outers

    def domain_names(self) -> List[str]:
        names = [self.interior_name]
        names.extend(p.domain_name("contact") for p in self.contacts)
        names.extend(p.domain_name("shank") for p in self.shanks)
        names.extend(p.domain_name("outer") for p in self.outers)
        if self.gauge is not None:
            names.append(self.gauge.domain_name("gauge"))
        return names


def outer_faces_from_box(box: Box3D, *, kind: str = "outer") -> List[PlanePatch]:
    x0, x1 = box.x
    y0, y1 = box.y
    z0, z1 = box.z
    return [
        PlanePatch("x_lo", "x", x0, {"y": (y0, y1), "z": (z0, z1)}, normal_sign=-1, kind=kind),
        PlanePatch("x_hi", "x", x1, {"y": (y0, y1), "z": (z0, z1)}, normal_sign=+1, kind=kind),
        PlanePatch("y_lo", "y", y0, {"x": (x0, x1), "z": (z0, z1)}, normal_sign=-1, kind=kind),
        PlanePatch("y_hi", "y", y1, {"x": (x0, x1), "z": (z0, z1)}, normal_sign=+1, kind=kind),
        PlanePatch("z_lo", "z", z0, {"x": (x0, x1), "y": (y0, y1)}, normal_sign=-1, kind=kind),
        PlanePatch("z_hi", "z", z1, {"x": (x0, x1), "y": (y0, y1)}, normal_sign=+1, kind=kind),
    ]


def make_gauge_patch(box: Box3D, *, size: float = 3.0, phi: float = 0.0) -> PlanePatch:
    half = float(size) * 0.5
    return PlanePatch(
        name="gauge",
        axis="z",
        value=box.z[1],
        span={"x": (-half, half), "y": (-half, half)},
        normal_sign=+1,
        kind="gauge",
        metadata={"bc_type": "dirichlet", "phi_V": float(phi)},
    )


def single_contact_reference(contact_name: str = "E0") -> TowerAGeometry:
    """Simple single-contact geometry for debugging and smoke tests."""

    box = Box3D(x=(-6.0, 6.0), y=(-6.0, 6.0), z=(-6.0, 6.0))
    z_hi = box.z[1]
    contact = PlanePatch(
        name=contact_name,
        axis="z",
        value=z_hi,
        span={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        normal_sign=+1,
        kind="contact",
    )
    outers = [p for p in outer_faces_from_box(box) if p.name != "z_hi"]
    gauge_patch = make_gauge_patch(box, size=2.0, phi=0.0)
    return TowerAGeometry(volume=box, contacts=[contact], shanks=[], outers=outers, gauge=gauge_patch)


def example_two_contacts() -> TowerAGeometry:
    """Small two-contact example for smoke tests and documentation."""

    box = Box3D(x=(-6.0, 6.0), y=(-6.0, 6.0), z=(-6.0, 6.0))
    z_hi = box.z[1]
    left = PlanePatch(
        name="E_left",
        axis="z",
        value=z_hi,
        span={"x": (-5.0, -3.0), "y": (-1.0, 1.0)},
        normal_sign=+1,
        kind="contact",
    )
    right = PlanePatch(
        name="E_right",
        axis="z",
        value=z_hi,
        span={"x": (3.0, 5.0), "y": (-1.0, 1.0)},
        normal_sign=+1,
        kind="contact",
    )
    outers = [p for p in outer_faces_from_box(box) if p.name != "z_hi"]
    gauge_patch = make_gauge_patch(box, size=2.0, phi=0.0)
    return TowerAGeometry(volume=box, contacts=[left, right], shanks=[], outers=outers, gauge=gauge_patch)


__all__ = [
    "Axis",
    "Box3D",
    "PlanePatch",
    "TowerAGeometry",
    "outer_faces_from_box",
    "make_gauge_patch",
    "single_contact_reference",
    "example_two_contacts",
]
