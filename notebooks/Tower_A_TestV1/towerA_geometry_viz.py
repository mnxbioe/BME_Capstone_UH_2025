"""
Quick geometry sanity test for Tower A.
Checks domains, patch normals, areas, and visualizes the setup.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bme_capstone.tower_a import Box3D, PlanePatch, TowerAGeometry, make_gauge_patch, outer_faces_from_box


def plot_patch(ax, patch, color, alpha=0.4, label=None):
    """Draw a rectangular patch in 3D."""
    span = patch.span
    ax1, ax2 = list(span.keys())
    s1 = [span[ax1][0], span[ax1][1]]
    s2 = [span[ax2][0], span[ax2][1]]

    X, Y = zip(*[(s1[0], s2[0]), (s1[1], s2[0]), (s1[1], s2[1]), (s1[0], s2[1])])

    verts = []
    if patch.axis == "x":
        verts = [[(patch.value, y, z) for y, z in zip(X, Y)]]
    elif patch.axis == "y":
        verts = [[(x, patch.value, z) for x, z in zip(X, Y)]]
    elif patch.axis == "z":
        verts = [[(x, y, patch.value) for x, y in zip(X, Y)]]

    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, label=label)
    ax.add_collection3d(poly)

def main():
    # --- 1. Define tissue volume
    box = Box3D(x=(-4, 4), y=(-4, 4), z=(-4, 4))
    print(f"Box extents (mm): {box.extents}")
    print(f"Box center (mm):  {box.center}")

    # --- 2. Define electrode patches
    contacts = [
        PlanePatch(
            name="E_left",
            axis="z",
            value=0.0,
            span={"x": (-1.0, -0.5), "y": (-0.3, 0.3)},
            normal_sign=+1,
            kind="contact"
        ),
        PlanePatch(
            name="E_right",
            axis="z",
            value=0.0,
            span={"x": (0.5, 1.0), "y": (-0.3, 0.3)},
            normal_sign=+1,
            kind="contact"
        ),
    ]

    # --- 3. Outer boundary faces (insulating walls)
    outers = outer_faces_from_box(box)
    outers.append(make_gauge_patch(box))  # add Dirichlet gauge

    # --- 4. Assemble geometry
    geom = TowerAGeometry(volume=box, contacts=contacts, outers=outers)

    # --- 5. Sanity checks
    print("\nRegistered domain keys:")
    for name in geom.domain_names():
        print("  ", name)

    print("\nPatch summary:")
    for p in geom.all_surfaces():
        print(f"{p.kind:>8s} | {p.name:>8s} | axis={p.axis} | area={p.area:.3f} | normal={p.normal}")

    # --- 6. Plot setup
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    # Volume outline
    x0, x1 = box.x; y0, y1 = box.y; z0, z1 = box.z
    for outer in outers:
        color = "lightgrey" if outer.kind == "outer" else "red"
        plot_patch(ax, outer, color=color, alpha=0.1)
    for contact in contacts:
        plot_patch(ax, contact, color="orange", alpha=0.8, label=contact.name)

    ax.set_title("Tower A Geometry Sanity Test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
