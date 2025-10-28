"""Geometry helpers for Tower A

These provide geometry descriptions that can be mapped
onto PINA domain objects. 

implementation focuses on 3-D box-shaped tissue volumes
axis-aligned plane patches representing
    - electrode contacts
    - insulating shanks
    - outer boundaries.  

Matches the Geometry Contract and maps to PINA's :class:`CartesianDomain` 

* :class:`Box3D` defines the padded tissue domain ``Omega``.
* :class:`PlanePatch` captures an axis-aligned planar patch (disc/square) together with its outward normal.
* :class:`TowerAGeometry` groups the interior domain and surface patches, and
  exposes helpers to build the dictionaries expected by a custom
  :class:`~pina.problem.SpatialProblem`.

The geometry objects themselves are agnostic to boundary-condition values.
Those are injected later when building the physics-informed problem.
"""

from __future__ import annotations # to avoid name error via a free reference classes that appear later in the file
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
from pina.domain import CartesianDomain #a PINA class that represents a region of space in (x, y, z).

# Coordinate safety net 
Axis = str  # restricted to {"x", "y", "z"}
_AXES: Tuple[Axis, Axis, Axis] = ("x", "y", "z")
def _validate_axis(axis: Axis) -> Axis:
    """Validate that ``axis`` is one of the three Cartesian axes."""
    if axis not in _AXES:
        raise ValueError(f"Expected axis in {_AXES}, got {axis!r}")
    return axis

#-- Verify simulation geometry is valid 
def _ensure_limits(bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Ensure bounds are ordered as (lo, hi) and non-degenerate."""
    lo, hi = map(float, bounds)
    if not lo < hi:
        raise ValueError(f"Bounds must satisfy lo < hi, got ({lo}, {hi})")
    return lo, hi


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned 3-D box used for the interior tissue domain.

    Parameters
    ----------
    x : tuple[float, float]
        Lower/upper bounds along the x axis (millimetres ).
    y : tuple[float, float]
        Lower/upper bounds along the y axis.
    z : tuple[float, float]
        Lower/upper bounds along the z axis.
    """

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    #to prevent invalid sampling / inverted domains
    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _ensure_limits(self.x))
        object.__setattr__(self, "y", _ensure_limits(self.y))
        object.__setattr__(self, "z", _ensure_limits(self.z))

    def to_domain(self) -> CartesianDomain:
        """Return the matching :class:`CartesianDomain`."""
        return CartesianDomain({"x": list(self.x), "y": list(self.y), "z": list(self.z)})
    
    #Compute the size of the box
    @property
    def extents(self) -> Tuple[float, float, float]:
        """Return the box edge lengths in each direction."""
        return (
            self.x[1] - self.x[0],
            self.y[1] - self.y[0],
            self.z[1] - self.z[0],
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        """geometric center (midpoint) of the box in 3-D space."""
        return (
            (self.x[0] + self.x[1]) * 0.5,
            (self.y[0] + self.y[1]) * 0.5,
            (self.z[0] + self.z[1]) * 0.5,
        )


@dataclass
class PlanePatch:
    """Axis-aligned(parallel) planar patch representing an electrode or boundary surface.

    The patch is described by fixing one axis to ``value`` and providing bounds
    along the remaining two axes through ``span``.

    Parameters
    ----------
    name : str
        Identifier used when registering:class:`TowerAGeometry`.

    axis : {"x", "y", "z"}
        Cartesian axis orthogonal to the patch

    value : float
        Coordinate value along ``axis`` where the plane lies.

    span : mapping[str, tuple[float, float]]
        Bounds for the remaining two axes. Keys must match the two axes other
        than ``axis``. Each tuple is interpreted as (lower, upper) bounds.

    normal_sign : int, optional
        Direction of the outward normal along ``axis``. ``+1`` denotes
        ``+axis`` and ``-1`` denotes ``-axis``. Defaults to ``+1``.
        
    kind : str, optional
        Semantic tag for later boundary-condition assignment. Suggested values
        include ``"contact"``, ``"shank"``, and ``"outer"``.  no
        effect on geometry

    metadata : dict, optional
        Free-form storage for additional geometric information (e.g., electrode
        channel index).
    """

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
            raise ValueError("normal_sign must be ±1")
        expected_other_axes = {ax for ax in _AXES if ax != self.axis}
        if set(self.span.keys()) != expected_other_axes:
            raise ValueError(
                f"span must provide bounds for axes {expected_other_axes}, "
                f"got keys={set(self.span.keys())}"
            )
        # Normalise bounds and ensure they are valid
        cleaned: Dict[Axis, Tuple[float, float]] = {}
        for ax, bounds in self.span.items():
            cleaned[ax] = _ensure_limits(bounds)
        object.__setattr__(self, "span", cleaned)

    #Calc area (used when converting electrode currents to flux densities)
    @property
    def area(self) -> float:
        """Return the area of the rectangular patch."""
        lengths = [bounds[1] - bounds[0] for bounds in self.span.values()]
        return lengths[0] * lengths[1] #area

    @property
    def normal(self) -> Tuple[float, float, float]:
        """Return the unit normal vector pointing outward from the tissue domain 
        (perpindicular to the plane)."""
        idx = _AXES.index(self.axis)
        vec = [0.0, 0.0, 0.0]
        vec[idx] = float(self.normal_sign)
        return tuple(vec)

    def to_domain(self) -> CartesianDomain:
        """Build the planar patch. which PINA uses to sample training points 
        along that plane(newman or Dic..)"""
        domain_dict = {self.axis: float(self.value)}
        for ax, bounds in self.span.items():
            domain_dict[ax] = [float(bounds[0]), float(bounds[1])]
        return CartesianDomain(domain_dict)

    def domain_name(self, prefix: Optional[str] = None) -> str:
        """Return a string identifer for patches 
        helps assemble `TowerAGeometry`.
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to the domain name. Defaults to ``self.kind``.
        """
        tag = prefix or self.kind
        return f"{tag}:{self.name}"

####################################################################################
# Assemble TowerAGeometry
@dataclass
class TowerAGeometry:
    """Container for the geometry
    ----------
    volume : Box3D
        Interior domain where the Laplace equation is enforced.
    contacts : list[PlanePatch]
        Planar patches representing stimulating/return contacts. 
         These are tagged with ``kind="contact"``.
    shanks : list[PlanePatch], (optional)
        Insulating shank surfaces (zero-flux).  
        Defaults to an empty list.
    outers : list[PlanePatch], 
        far-field boundaries (Dirichlet or zero-flux).
    interior_name : str, (optional)
        Dictionary key used for the interior domain. Defaults to ``"interior"``.
    """

    volume: Box3D
    contacts: List[PlanePatch]
    shanks: List[PlanePatch] = field(default_factory=list)
    outers: List[PlanePatch] = field(default_factory=list)
    interior_name: str = "interior"

    def build_domains(self) -> Dict[str, CartesianDomain]:
        """Return the dictionary of domains expected by a PINA problem."""
        domains: Dict[str, CartesianDomain] = {
            self.interior_name: self.volume.to_domain() #first line defines the main domain Ω
        }
        #Each for loop adds a different kind of surface to the same dictionary, 
        # using a prefix ("contact:", "shank:", "outer:")
        # to make it easy to tell them apart later.
        for patch in self.contacts:
            domains[patch.domain_name("contact")] = patch.to_domain()
        for patch in self.shanks:
            domains[patch.domain_name("shank")] = patch.to_domain()
        for patch in self.outers:
            domains[patch.domain_name("outer")] = patch.to_domain()
        return domains
    
        """TowerAGeometry
            ├── volume: Box3D
            │     └── → 1 interior CartesianDomain
            ├── contacts: [PlanePatch, PlanePatch, ...]
            │     └── → contact:E1, contact:E2, ...
            ├── shanks: [PlanePatch, ...]
            │     └── → shank:S1, shank:S2, ...
            └── outers: [PlanePatch, ...]
                └── → outer:top, outer:bottom, ..."""

    # loop through different groups of surfaces (contacts, shanks, outers) 
    # without having to know how they’re stored internally.:
    def iter_contacts(self) -> Iterable[PlanePatch]:
        """Iterate over electrode contact surfaces."""
        return iter(self.contacts)

    def iter_shanks(self) -> Iterable[PlanePatch]:
        """Iterate over insulating shank surfaces."""
        return iter(self.shanks)

    def iter_outers(self) -> Iterable[PlanePatch]:
        """Iterate over outer-domain surfaces."""
        return iter(self.outers)

    def all_surfaces(self) -> Iterable[PlanePatch]:
        """Iterate over every registered planar patch."""
        yield from self.contacts
        yield from self.shanks
        yield from self.outers

    def domain_names(self) -> List[str]:
        """List all domain keys (including interior) produced by ``build_domains``."""
        names = [self.interior_name]
        names.extend(p.domain_name("contact") for p in self.contacts)
        names.extend(p.domain_name("shank") for p in self.shanks)
        names.extend(p.domain_name("outer") for p in self.outers)
        return names
# new util (or add to geometry.py)
from bme_capstone.tower_a.geometry import PlanePatch, Box3D, TowerAGeometry

def outer_faces_from_box(box: Box3D, pad_kind="outer"):
    x0, x1 = box.x; y0, y1 = box.y; z0, z1 = box.z
    return [
        PlanePatch(name="x_lo", axis="x", value=x0, span={"y": (y0,y1), "z": (z0,z1)}, normal_sign=-1, kind=pad_kind),
        PlanePatch(name="x_hi", axis="x", value=x1, span={"y": (y0,y1), "z": (z0,z1)}, normal_sign=+1, kind=pad_kind),
        PlanePatch(name="y_lo", axis="y", value=y0, span={"x": (x0,x1), "z": (z0,z1)}, normal_sign=-1, kind=pad_kind),
        PlanePatch(name="y_hi", axis="y", value=y1, span={"x": (x0,x1), "z": (z0,z1)}, normal_sign=+1, kind=pad_kind),
        PlanePatch(name="z_lo", axis="z", value=z0, span={"x": (x0,x1), "y": (y0,y1)}, normal_sign=-1, kind=pad_kind),
        PlanePatch(name="z_hi", axis="z", value=z1, span={"x": (x0,x1), "y": (y0,y1)}, normal_sign=+1, kind=pad_kind),
    ]
