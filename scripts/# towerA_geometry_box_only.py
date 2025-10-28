# towerA_geometry_electrodes_3d_2x8.py
# Run:  python towerA_geometry_electrodes_3d_2x8.py
# Deps: numpy, matplotlib

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Geometry config (units: mm)
# ----------------------------
@dataclass
class Domain:
    xmin: float = -6.0
    xmax: float =  6.0
    ymin: float = -6.0
    ymax: float =  6.0
    zmin: float = -6.0
    zmax: float =  6.0
    array_plane_z: float = 0.0  # z_a; +z points into tissue

@dataclass
class ElectrodeSpec:
    name: str
    center_xy: Tuple[float, float]        # (x, y) on array plane
    tip_radius_mm: float = 0.025          # hemisphere radius (25 μm)
    shank_radius_mm: float = 0.05         # cylinder radius (50 μm)
    shank_length_mm: float = 2.0          # cylinder length into +z
    current_sign: int = +1                # +1 arrow into +z, -1 arrow toward -z

D = Domain()

def build_grid_2x8(pitch_mm=0.40,
                   tip_r=0.025,
                   shank_r=0.05,
                   shank_len=2.0) -> List[ElectrodeSpec]:
    """Centers a 2×8 grid at (0,0) on z=0. Rows along +y, columns along +x."""
    M, N = 2, 8
    xs = [ (i - (N-1)/2)*pitch_mm for i in range(N) ]
    ys = [ (j - (M-1)/2)*pitch_mm for j in range(M) ]
    specs: List[ElectrodeSpec] = []
    for j, y in enumerate(ys, start=1):
        for i, x in enumerate(xs, start=1):
            name = f"E{j:02d}{i:02d}"
            # simple alternating current sign by column (visual arrow)
            sgn = +1 if (i % 2 == 1) else -1
            specs.append(ElectrodeSpec(
                name=name,
                center_xy=(x, y),
                tip_radius_mm=tip_r,
                shank_radius_mm=shank_r,
                shank_length_mm=shank_len,
                current_sign=sgn
            ))
    return specs

ELECTRODES = build_grid_2x8()

# ----------------------------
# 2) 3D primitives (meshes)
# ----------------------------
def hemisphere_mesh(center=(0,0,0), radius=0.05, n_theta=32, n_phi=24):
    """
    Hemisphere oriented along +z with equator on z=0 plane.
    center: (x0,y0,0) is equator center; hemisphere bulges into +z.
    """
    x0, y0, z0 = center
    # phi: 0..pi/2 (0 at "north pole" would be odd; we want equator at 0)
    # We'll param with elevation from 0 (equator) to +pi/2 (top)
    phi = np.linspace(0, np.pi/2, n_phi)          # elevation
    theta = np.linspace(0, 2*np.pi, n_theta)      # azimuth
    Phi, Theta = np.meshgrid(phi, theta, indexing="ij")
    R = radius
    # Equator at z=z0, bulging into +z
    X = x0 + R*np.cos(Theta)*np.cos(Phi)
    Y = y0 + R*np.sin(Theta)*np.cos(Phi)
    Z = z0 + R*np.sin(Phi)
    return X, Y, Z

def cylinder_mesh(center=(0,0,0), radius=0.05, z0=0.0, z1=2.0, n_theta=48, n_z=20):
    """
    Straight cylinder aligned with +z from z0 to z1, centered at (x0,y0).
    """
    x0, y0, _ = center
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z0, z1, n_z)
    Theta, Z = np.meshgrid(theta, z, indexing="ij")
    X = x0 + radius*np.cos(Theta)
    Y = y0 + radius*np.sin(Theta)
    return X, Y, Z

def equalize_3d(ax):
    # Equal aspect ratio for 3D
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    spans = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    maxspan = spans.max()
    cx, cy, cz = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim(cx - maxspan/2, cx + maxspan/2)
    ax.set_ylim(cy - maxspan/2, cy + maxspan/2)
    ax.set_zlim(cz - maxspan/2, cz + maxspan/2)

# ----------------------------
# 3) Plot: Ω (wireframe) + full 3D electrodes
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Ω wireframe (faces as transparent quads)
faces = [
    [(D.xmax,D.ymin,D.zmin),(D.xmax,D.ymax,D.zmin),(D.xmax,D.ymax,D.zmax),(D.xmax,D.ymin,D.zmax)],  # +X
    [(D.xmin,D.ymin,D.zmin),(D.xmin,D.ymin,D.zmax),(D.xmin,D.ymax,D.zmax),(D.xmin,D.ymax,D.zmin)],  # -X
    [(D.xmin,D.ymax,D.zmin),(D.xmax,D.ymax,D.zmin),(D.xmax,D.ymax,D.zmax),(D.xmin,D.ymax,D.zmax)],  # +Y
    [(D.xmin,D.ymin,D.zmin),(D.xmin,D.ymin,D.zmax),(D.xmax,D.ymin,D.zmax),(D.xmax,D.ymin,D.zmin)],  # -Y
    [(D.xmin,D.ymin,D.zmax),(D.xmax,D.ymin,D.zmax),(D.xmax,D.ymax,D.zmax),(D.xmin,D.ymax,D.zmax)],  # +Z
    [(D.xmin,D.ymin,D.zmin),(D.xmin,D.ymax,D.zmin),(D.xmax,D.ymax,D.zmin),(D.xmax,D.ymin,D.zmin)],  # -Z
]
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
for f in faces:
    ax.add_collection3d(Poly3DCollection([f], alpha=0.07, edgecolor="k", linewidths=0.8))

# Array plane line grid for context (thin, optional)
show_array_plane_grid = True
if show_array_plane_grid:
    z = D.array_plane_z
    xs = np.linspace(D.xmin, D.xmax, 5)
    ys = np.linspace(D.ymin, D.ymax, 5)
    for x in xs:
        ax.plot([x,x],[D.ymin,D.ymax],[z,z], linewidth=0.4)
    for y in ys:
        ax.plot([D.xmin,D.xmax],[y,y],[z,z], linewidth=0.4)

# Draw each electrode: hemisphere tip @ z=0 + cylinder to z = shank_len
for e in ELECTRODES:
    x0, y0 = e.center_xy
    z_eq = D.array_plane_z  # hemisphere equator and cylinder base at z=0

    # Hemisphere (bulging into +z)
    Xh, Yh, Zh = hemisphere_mesh(center=(x0, y0, z_eq), radius=e.tip_radius_mm,
                                 n_theta=48, n_phi=36)
    ax.plot_surface(Xh, Yh, Zh, alpha=0.9, rstride=1, cstride=1, linewidth=0)

    # Cylinder (from z = tip_radius to z = tip_radius + shank_length)
    z0 = z_eq + e.tip_radius_mm
    z1 = z_eq + e.tip_radius_mm + e.shank_length_mm
    Xc, Yc, Zc = cylinder_mesh(center=(x0, y0, z_eq),
                               radius=e.shank_radius_mm, z0=z0, z1=z1,
                               n_theta=64, n_z=40)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.8, rstride=1, cstride=1, linewidth=0)

    # Current direction arrow: +z for +1, -z for -1 (purely visual)
    arrow_len = 0.5  # mm
    sgn = 1.0 if e.current_sign >= 0 else -1.0
    ax.quiver(x0, y0, z_eq + e.tip_radius_mm*0.6, 0, 0, sgn,
              length=arrow_len, arrow_length_ratio=0.2, linewidth=1.6)

    # Label
    ax.text(x0, y0, z1 + 0.1, e.name, fontsize=7, ha="center", va="bottom")

# Axes & view
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
ax.set_title("Tower A · Full 3D Electrodes (2×8) — hemispherical tips + cylindrical shanks")

ax.set_xlim(D.xmin, D.xmax)
ax.set_ylim(D.ymin, D.ymax)
ax.set_zlim(D.zmin, D.zmax)
equalize_3d(ax)
ax.view_init(elev=22, azim=40)  # a nice default view
plt.tight_layout()
plt.show()
