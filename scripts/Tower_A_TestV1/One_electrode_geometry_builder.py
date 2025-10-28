"""Single‑electrode Tower A geometry builder with plotting and sanity checks.

all values are defined below 

The geometry is ready to be fed into the trainer
later (returns a TowerAGeometry and builds PINA domains for checks).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Tuple
from pathlib import Path

# Ensure the package modules resolve when running from scripts/
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from bme_capstone.tower_a import Box3D, PlanePatch, TowerAGeometry  # noqa: E402


# -----------------------------
# Editable geometry parameters
# -----------------------------
# Tissue volume (mm)
DOMAIN_X: Tuple[float, float] = (-4.0, 4.0)
DOMAIN_Y: Tuple[float, float] = (-4.0, 4.0)
DOMAIN_Z: Tuple[float, float] = (-4.0, 4.0)

# Single square contact on z=0 (mm). Size = 0.4 x 0.4 mm centered at (0, 0)
CONTACT_NAME = "E1"
CONTACT_AXIS = "z"
CONTACT_Z = 0.0
CONTACT_HALF_SIZE = 0.2  # -> 0.4 mm square
CONTACT_CENTER_X = 0.0
CONTACT_CENTER_Y = 0.0

# Gauge (Dirichlet) reference patch on +Z. Square 3 x 3 mm centered at (0, 0)
GAUGE_HALF_SIZE = 1.5
GAUGE_PHI_V = 0.0


def build_geometry() -> TowerAGeometry:
    """Construct and return the Tower A geometry for a single square contact."""
    volume = Box3D(x=DOMAIN_X, y=DOMAIN_Y, z=DOMAIN_Z)

    contacts = [
        PlanePatch(
            name=CONTACT_NAME,
            axis=CONTACT_AXIS,
            value=float(CONTACT_Z),
            span={
                "x": (
                    CONTACT_CENTER_X - CONTACT_HALF_SIZE,
                    CONTACT_CENTER_X + CONTACT_HALF_SIZE,
                ),
                "y": (
                    CONTACT_CENTER_Y - CONTACT_HALF_SIZE,
                    CONTACT_CENTER_Y + CONTACT_HALF_SIZE,
                ),
            },
            normal_sign=+1,  # outward +z
            kind="contact",
        ),
    ]

    # Outer faces (full faces). Low side normals -1, high side +1
    x0, x1 = DOMAIN_X
    y0, y1 = DOMAIN_Y
    z0, z1 = DOMAIN_Z
    outers = [
        PlanePatch(name="x_lo", axis="x", value=x0, span={"y": (y0, y1), "z": (z0, z1)}, normal_sign=-1, kind="outer"),
        PlanePatch(name="x_hi", axis="x", value=x1, span={"y": (y0, y1), "z": (z0, z1)}, normal_sign=+1, kind="outer"),
        PlanePatch(name="y_lo", axis="y", value=y0, span={"x": (x0, x1), "z": (z0, z1)}, normal_sign=-1, kind="outer"),
        PlanePatch(name="y_hi", axis="y", value=y1, span={"x": (x0, x1), "z": (z0, z1)}, normal_sign=+1, kind="outer"),
        PlanePatch(name="z_lo", axis="z", value=z0, span={"x": (x0, x1), "y": (y0, y1)}, normal_sign=-1, kind="outer"),
    ]

    # Gauge patch on +Z (square). Tag with Dirichlet metadata so trainer can pick it up later.
    gauge = PlanePatch(
        name="gauge",
        axis="z",
        value=z1,
        span={"x": (-GAUGE_HALF_SIZE, GAUGE_HALF_SIZE), "y": (-GAUGE_HALF_SIZE, GAUGE_HALF_SIZE)},
        normal_sign=+1,
        kind="outer",
        metadata={"bc_type": "dirichlet", "phi_V": GAUGE_PHI_V},
    )
    outers.append(gauge)

    return TowerAGeometry(volume=volume, contacts=contacts, shanks=[], outers=outers)


def describe_geometry(geometry: TowerAGeometry) -> None:
    print("Interior volume (mm):")
    print(f"  x: {geometry.volume.x}")
    print(f"  y: {geometry.volume.y}")
    print(f"  z: {geometry.volume.z}")
    print("\nContacts:")
    for p in geometry.iter_contacts():
        span_str = ", ".join(f"{ax}=({lo:.3f}, {hi:.3f})" for ax, (lo, hi) in p.span.items())
        print(f"  {p.name}: axis={p.axis} z={p.value:+.3f} area={p.area:.3f} mm^2 normal={p.normal} span[{span_str}]")
    print("\nOuter surfaces:")
    for p in geometry.iter_outers():
        bc = p.metadata.get("bc_type") if isinstance(p.metadata, Mapping) else None
        span_str = ", ".join(f"{ax}=({lo:.3f}, {hi:.3f})" for ax, (lo, hi) in p.span.items())
        extra = f" bc={bc}" if bc else ""
        print(f"  {p.name}: axis={p.axis} v={p.value:+.3f} area={p.area:.3f} mm^2 normal={p.normal} span[{span_str}]{extra}")


def _plot_patch(ax, patch: PlanePatch, color: str, alpha: float = 0.4, label: str | None = None) -> None:
    span = patch.span
    ax1, ax2 = list(span.keys())
    s1 = [span[ax1][0], span[ax1][1]]
    s2 = [span[ax2][0], span[ax2][1]]
    X, Y = list(zip(*[(s1[0], s2[0]), (s1[1], s2[0]), (s1[1], s2[1]), (s1[0], s2[1])]))
    if patch.axis == "x":
        verts = [[(patch.value, y, z) for y, z in zip(X, Y)]]
    elif patch.axis == "y":
        verts = [[(x, patch.value, z) for x, z in zip(X, Y)]]
    else:
        verts = [[(x, y, patch.value) for x, y in zip(X, Y)]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, label=label)
    ax.add_collection3d(poly)


def plot_geometry(geometry: TowerAGeometry, backend: str = "TkAgg") -> None:
    try:
        import matplotlib
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting not available ({exc}). Install matplotlib or change backend.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    for outer in geometry.iter_outers():
        is_gauge = (isinstance(outer.metadata, Mapping) and outer.metadata.get("bc_type") == "dirichlet") or outer.name.lower() == "gauge"
        color = "deepskyblue" if is_gauge else "grey"
        _plot_patch(ax, outer, color=color, alpha=0.5)
    for contact in geometry.iter_contacts():
        _plot_patch(ax, contact, color="orange", alpha=0.8, label=contact.name)

    x0, x1 = geometry.volume.x; y0, y1 = geometry.volume.y; z0, z1 = geometry.volume.z
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_title("Tower A: Single-Electrode Geometry")
    try:
        plt.legend()
    except Exception:
        pass
    plt.tight_layout(); plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct single-electrode geometry build + plot")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.set_defaults(plot=True)
    parser.add_argument("--matplotlib-backend", default="TkAgg")
    args = parser.parse_args()

    geometry = build_geometry()

    # Ready for trainer: ensure PINA domains build and list keys
    domains = geometry.build_domains()
    print("\nRegistered domain keys:")
    for k in domains.keys():
        print("  ", k)
    print()
    describe_geometry(geometry)

    if args.plot:
        plot_geometry(geometry, backend=args.matplotlib_backend)


if __name__ == "__main__":
    main()
