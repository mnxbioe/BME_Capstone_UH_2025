# =============================================================================
# TOWER A v2 GEOMETRY MODULE
# =============================================================================

# logical construction process:
# define primitives → assemble boundaries → compose full geometry → export.

# Module defines the geometric specification for Tower A (the field encoder).
# The file is organized with the following sections:
#   1. Imports and utility functions 
#   2. Core geometry primitives — Box3D (volume) and PlanePatch (surface).
#   3. TowerAGeometry container — groups volume and patches into domains.
#   4. Utility builders — helper functions for common boundary elements.
#   5. Single-contact geometry — for inital testing/debugging.
#   6. Grid-contact geometry — final array configuration (MxN).
#   7. Module exports — symbols available for import elsewhere.
# =============================================================================

# %% ---------------------------------------------------------------------
# SECTION 1 — Imports and utility functions
# ---------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
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

# %% ---------------------------------------------------------------------
# SECTION 2 — Core geometry primitives (Box3D, PlanePatch)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Box3D:
    """Axis-aligned box describing the tissue volume (mm)
     Represents the interior domain Ω."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    # --- Validation -------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _ensure_limits(self.x))
        object.__setattr__(self, "y", _ensure_limits(self.y))
        object.__setattr__(self, "z", _ensure_limits(self.z))

    # --- Conversions -------------------------------------------------
    def to_domain(self) -> CartesianDomain:
         #Convert the box into a PINA CartesianDomain for PDE sampling
        return CartesianDomain({"x": list(self.x), "y": list(self.y), "z": list(self.z)})
     
     # --- Derived properties ---------------------------------------------
    @property
    def extents(self) -> Tuple[float, float, float]:
        """Return side lengths Δx, Δy, Δz (useful for scaling and diagnostics)."""
        return (
            self.x[1] - self.x[0],
            self.y[1] - self.y[0],
            self.z[1] - self.z[0],
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the geometric center of the box (midpoint of each axis)."""
        return (
            (self.x[0] + self.x[1]) * 0.5,
            (self.y[0] + self.y[1]) * 0.5,
            (self.z[0] + self.z[1]) * 0.5,
        )

@dataclass
class PlanePatch:
    """ Each patch represents a 2-D subset of the domain boundary
        • electrode contacts (Neumann flux boundary)
        • insulating shanks (zero-flux boundary)
        • outer walls (Dirichlet or Neumann BCs)
        • gauge patch (reference potential)

    Together, PlanePatch instances define all boundary conditions
    applied to the PDE inside the Box3D domain.
    """
    # --- Geometric and logical identifiers --------------------------
    name: str
    axis: Axis
    value: float
    span: Mapping[Axis, Tuple[float, float]]
    normal_sign: int = 1  # +1 → outward normal, -1 → inward normal
    kind: str = "contact"
    metadata: MutableMapping[str, float] = field(default_factory=dict)

    # --- Validation ---------------------------------------------------
    def __post_init__(self) -> None:
        self.axis = _validate_axis(self.axis)
        if self.normal_sign not in (-1, 1):  
            raise ValueError("normal_sign must be -1 or +1")
        expected_other_axes = {ax for ax in _AXES if ax != self.axis}
        cleaned: Dict[Axis, Tuple[float, float]] = {}
        for ax, bounds in self.span.items():
            cleaned[ax] = _ensure_limits(bounds)
        object.__setattr__(self, "span", cleaned)

    # --- Derived properties ----------------------------------------
    @property #Simple rectangle area from the two span lengths
    def area(self) -> float: 
        lengths = [bounds[1] - bounds[0] for bounds in self.span.values()]
        return lengths[0] * lengths[1]

    @property #unit normal vector
    def normal(self) -> Tuple[float, float, float]:
        vec = [0.0, 0.0, 0.0]
        vec[_AXES.index(self.axis)] = float(self.normal_sign)
        return tuple(vec)  # type: ignore[return-value]
    
    # --- Domain conversion ----------------------------------------
    def to_domain(self) -> CartesianDomain:
        """
        Convert this PlanePatch into a PINA CartesianDomain.
        This allows PINA to sample boundary points and apply boundary
        conditions on this specific rectangular surface.
        """
        payload = {self.axis: float(self.value)}
        for ax, bounds in self.span.items():
            payload[ax] = [float(bounds[0]), float(bounds[1])]
        return CartesianDomain(payload)

    def domain_name(self, prefix: Optional[str] = None) -> str:
        tag = prefix or self.kind
        return f"{tag}:{self.name}"

# %% ---------------------------------------------------------------------
# SECTION 3: Tower A Geometry Container (Domain Builder)
# ---------------------------------------------------------------------
# group together the 3-D tissue volume (Box3D) and all surface patches
# (contacts, shanks, outers, gauge). From these:
#   • Build a full dictionary of PINA CartesianDomains for PDE training.
#   • Provide iterators over specific surface types.
#   • Return consistent domain names for boundary condition assignment.
@dataclass
class TowerAGeometry:
    """Group the interior volume and planar surface patches."""

    volume: Box3D
    contacts: List[PlanePatch]
    shanks: List[PlanePatch] = field(default_factory=list)
    outers: List[PlanePatch] = field(default_factory=list)
    interior_name: str = "interior"
    gauge: Optional[PlanePatch] = None

    # --- Domain assembly ---------------------------------------------
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
    

# %% ---------------------------------------------------------------------
# SECTION 4 — Utility builders (outer faces, gauge patch, rectangles)
# ---------------------------------------------------------------------
# These help  generate sets of PlanePatch objects that make up
#  boundary elements of Tower A geometries.
#
#   • outer_faces_from_box() → returns the six external faces of a Box3D.
#   • make_gauge_patch()     → builds a square Dirichlet “ground” patch.
#   • _rectangles_around_square() → produces the insulating shank regions
#                                   surrounding one or more contact openings.
# Each returns PlanePatch instances ready to be passed to TowerAGeometry.
# ---------------------------------------------------------------------

# --- 4.1  Outer-face builder ---------------------------------
def outer_faces_from_box(box: Box3D, *, kind: str = "outer") -> List[PlanePatch]:
    x0, x1 = box.x
    y0, y1 = box.y
    z0, z1 = box.z
    # Construct each face with its outward normal orientation
    return [
        PlanePatch("x_lo", "x", x0, {"y": (y0, y1), "z": (z0, z1)}, normal_sign=-1, kind=kind),
        PlanePatch("x_hi", "x", x1, {"y": (y0, y1), "z": (z0, z1)}, normal_sign=+1, kind=kind),
        PlanePatch("y_lo", "y", y0, {"x": (x0, x1), "z": (z0, z1)}, normal_sign=-1, kind=kind),
        PlanePatch("y_hi", "y", y1, {"x": (x0, x1), "z": (z0, z1)}, normal_sign=+1, kind=kind),
        PlanePatch("z_lo", "z", z0, {"x": (x0, x1), "y": (y0, y1)}, normal_sign=-1, kind=kind),
        PlanePatch("z_hi", "z", z1, {"x": (x0, x1), "y": (y0, y1)}, normal_sign=+1, kind=kind),
    ]

 #4.2  Gauge-patch builder -----------------------------------
def make_gauge_patch(
    box: Box3D,
    *,
    size: float = 3.0,
    phi: float = 0.0,
    center: Tuple[float, float] | None = None,
) -> PlanePatch:
    """
    Create a square Dirichlet patch/gauge on the top surface of the box.

    Parameters
    ----------
    size : Edge length of the square patch (mm).
    phi : potential value assigned to this patch (volts).
    center : Optional (x, y) coordinates of the patch center; defaults to (0, 0).

    Returns
    -------
    PlanePatch
        Patch tagged as 'gauge', located on the top (z_hi) face.
    """
    # Compute half-size and center location
    half = float(size) * 0.5
    if center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = center

    # Build the gauge surface aligned with the top of the domain
    return PlanePatch(
        name="gauge",
        axis="z",
        value=box.z[1],
        span={"x": (cx - half, cx + half), "y": (cy - half, cy + half)},
        normal_sign=+1,
        kind="gauge",
        metadata={"bc_type": "dirichlet", "phi_V": float(phi)},
    )

# --- 4.3  Rectangle-tiling helper ---------------------------------
def _rectangles_around_square(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    square_x: Tuple[float, float],
    square_y: Tuple[float, float],
    *,
    axis: Axis,
    value: float,
    name_prefix: str,
    normal_sign: int,
    kind: str,
) -> List[PlanePatch]:
    """
    Generate four rectangular PlanePatches that surround an inner square.

    Used to fill the remaining surface area of a plane after cutting out
    a smaller square region (e.g., a contact opening or gauge window).

    Returns
    -------
    List[PlanePatch]
        Rectangles for the left, right, bottom, and top regions.
    """
    # --- Unpack bounding coordinates ----------------------------
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    sq_x0, sq_x1 = square_x
    sq_y0, sq_y1 = square_y

    patches: List[PlanePatch] = []

    def _add(name: str, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> None:
        if x_hi <= x_lo or y_hi <= y_lo:
            return
        patches.append(
            PlanePatch(
                name,
                axis,
                value,
                {"x": (x_lo, x_hi), "y": (y_lo, y_hi)},
                normal_sign=normal_sign,
                kind=kind,
            )
        )
    # --- Define four surrounding rectangles -----------------------
    _add(f"{name_prefix}_left", x0, sq_x0, y0, y1)
    _add(f"{name_prefix}_right", sq_x1, x1, y0, y1)
    _add(f"{name_prefix}_bottom", sq_x0, sq_x1, y0, sq_y0)
    _add(f"{name_prefix}_top", sq_x0, sq_x1, sq_y1, y1)
    return patches

# %% ---------------------------------------------------------------------
# SECTION 5 — Geometry single-contact ( For Testing)
# ---------------------------------------------------------------------
def single_contact_geometry(
    *,
    domain_xy_mm: float = 12.0,
    domain_depth_mm: float = 12.0,
    array_plane_z: float = 0.0,
    contact_radius_mm: float = 0.025,
    gauge_radius_mm: float = 1.5,
) -> TowerAGeometry:
    """Tower A contract L0 single-contact geometry."""

    half_xy = domain_xy_mm * 0.5
    z_lo = array_plane_z
    z_hi = array_plane_z + domain_depth_mm
    box = Box3D(x=(-half_xy, half_xy), y=(-half_xy, half_xy), z=(z_lo, z_hi))

    half_contact = float(contact_radius_mm)
    contact = PlanePatch(
        name="E01",
        axis="z",
        value=array_plane_z,
        span={"x": (-half_contact, half_contact), "y": (-half_contact, half_contact)},
        normal_sign=-1,
        kind="contact",
        metadata={"radius_mm": contact_radius_mm},
    )

    # Insulating resin/shank covering the remainder of the array plane
    shank_patches = _rectangles_around_square(
        box.x,
        box.y,
        contact.span["x"],
        contact.span["y"],
        axis="z",
        value=array_plane_z,
        name_prefix="shank_z_lo",
        normal_sign=-1,
        kind="shank",
    )

    gauge_patch = make_gauge_patch(box, size=gauge_radius_mm * 2.0, phi=0.0, center=(0.0, 0.0))

    outers: List[PlanePatch] = []
    for patch in outer_faces_from_box(box):
        if patch.name == "z_lo":
            continue  # handled by contact + shank patches
        if patch.name == "z_hi":
            outers.extend(
                _rectangles_around_square(
                    box.x,
                    box.y,
                    gauge_patch.span["x"],
                    gauge_patch.span["y"],
                    axis="z",
                    value=z_hi,
                    name_prefix="z_hi_outer",
                    normal_sign=+1,
                    kind="outer",
                )
            )
        else:
            outers.append(patch)

    return TowerAGeometry(volume=box, contacts=[contact], shanks=shank_patches, outers=outers, gauge=gauge_patch)

# %% ---------------------------------------------------------------------
# SECTION 6 — Geometry presets grid arrays (final Verson)
# ---------------------------------------------------------------------
def grid_contact_geometry(
    rows: int = 2,
    cols: int = 8,
    *,
    domain_xy_mm: float = 12.0,
    domain_depth_mm: float = 12.0,
    array_plane_z: float = 0.0,
    pitch_mm: float = 0.40,
    contact_radius_mm: float = 0.025,
    gauge_radius_mm: float = 1.5,
    active_contacts: Optional[Sequence[str]] = None,
) -> TowerAGeometry:
    """Tower A contract L1 grid (MxN) geometry."""

    half_xy = domain_xy_mm * 0.5
    z_lo = array_plane_z
    z_hi = array_plane_z + domain_depth_mm
    box = Box3D(x=(-half_xy, half_xy), y=(-half_xy, half_xy), z=(z_lo, z_hi))

    def contact_name(r: int, c: int) -> str:
        return f"E{r:02d}{c:02d}"

    half_pitch_x = (cols - 1) * 0.5 * pitch_mm
    half_pitch_y = (rows - 1) * 0.5 * pitch_mm
    half_contact = float(contact_radius_mm)

    contacts: List[PlanePatch] = []
    for r in range(rows):
        y = (half_pitch_y - r * pitch_mm)
        for c in range(cols):
            x = (-half_pitch_x + c * pitch_mm)
            name = contact_name(r + 1, c + 1)
            if active_contacts and name not in active_contacts:
                continue
            contacts.append(
                PlanePatch(
                    name,
                    axis="z",
                    value=array_plane_z,
                    span={"x": (x - half_contact, x + half_contact), "y": (y - half_contact, y + half_contact)},
                    normal_sign=-1,
                    kind="contact",
                    metadata={"radius_mm": contact_radius_mm, "pitch_mm": pitch_mm},
                )
            )

    if not contacts:
        raise ValueError("No contacts selected; check active_contacts parameter.")

    # Compute shank patches by subtracting the bounding rectangle of all contacts
    min_x = min(p.span["x"][0] for p in contacts)
    max_x = max(p.span["x"][1] for p in contacts)
    min_y = min(p.span["y"][0] for p in contacts)
    max_y = max(p.span["y"][1] for p in contacts)
    shank_patches = _rectangles_around_square(
        box.x,
        box.y,
        (min_x, max_x),
        (min_y, max_y),
        axis="z",
        value=array_plane_z,
        name_prefix="shank_z_lo",
        normal_sign=-1,
        kind="shank",
    )

    gauge_patch = make_gauge_patch(box, size=gauge_radius_mm * 2.0, phi=0.0, center=(0.0, 0.0))

    outers: List[PlanePatch] = []
    for patch in outer_faces_from_box(box):
        if patch.name == "z_lo":
            continue
        if patch.name == "z_hi":
            outers.extend(
                _rectangles_around_square(
                    box.x,
                    box.y,
                    gauge_patch.span["x"],
                    gauge_patch.span["y"],
                    axis="z",
                    value=z_hi,
                    name_prefix="z_hi_outer",
                    normal_sign=+1,
                    kind="outer",
                )
            )
        else:
            outers.append(patch)

    return TowerAGeometry(volume=box, contacts=contacts, shanks=shank_patches, outers=outers, gauge=gauge_patch)

# %% ---------------------------------------------------------------------
# SECTION 7 — Module exports
# ----------------------------------------------------------
__all__ = [
    "Axis",
    "Box3D",
    "PlanePatch",
    "TowerAGeometry",
    "outer_faces_from_box",
    "make_gauge_patch",
    "single_contact_geometry",
    "grid_contact_geometry",
]
