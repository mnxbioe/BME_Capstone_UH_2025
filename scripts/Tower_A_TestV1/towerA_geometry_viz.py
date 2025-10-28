# towerA_geometry_viz_2x8.py
# Run:  python towerA_geometry_viz_2x8.py
# Deps: numpy, matplotlib

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------
# 1) Geometry config (12 mm box)
# ----------------------------
@dataclass
class Domain:
    xmin: float = -6.0
    xmax: float =  6.0
    ymin: float = -6.0
    ymax: float =  6.0
    zmin: float = -6.0
    zmax: float =  6.0
    array_plane_z: float = 0.0  # z_a

@dataclass
class Electrode:
    name: str
    center: Tuple[float, float, float]  # (x,y,z) in mm
    radius_mm: float = 0.025            # 25 μm
    bc: str = "Neumann"
    current: float = +1.0               # + injects into tissue (+z)

@dataclass
class ReferencePatch:
    face: str = "+Z"     # Γ₀ on the +Z outer face
    center: Tuple[float,float,float] = (0.0,0.0,6.0)
    radius_mm: float = 1.5
    phi_V: float = 0.0

D = Domain()
GAUGE = ReferencePatch()

# ----------------------------
# 2) Build the 2×8 array (Γₑ)
# ----------------------------
def build_grid_electrodes(M=2, N=8, pitch_mm=0.40, radius_mm=0.025) -> List[Electrode]:
    """
    Centers an MxN grid at (0,0,z_a). Rows along +y, columns along +x.
    Names: E(row)(col) -> E0101..E0208
    """
    xs = [ (i - (N-1)/2)*pitch_mm for i in range(N) ]
    ys = [ (j - (M-1)/2)*pitch_mm for j in range(M) ]
    electrodes: List[Electrode] = []
    for j, y in enumerate(ys, start=1):
        for i, x in enumerate(xs, start=1):
            name = f"E{j:02d}{i:02d}"
            center = (x, y, D.array_plane_z)
            # simple bipolar-like pattern: alternate sign by column
            current = +1.0 if (i % 2 == 1) else -1.0
            electrodes.append(Electrode(name=name, center=center,
                                        radius_mm=radius_mm, bc="Neumann",
                                        current=current))
    return electrodes

ELECTRODES = build_grid_electrodes(M=2, N=8, pitch_mm=0.40, radius_mm=0.025)

# ----------------------------
# 3) Helpers
# ----------------------------
def box_faces(d: Domain):
    x0, x1 = d.xmin, d.xmax
    y0, y1 = d.ymin, d.ymax
    z0, z1 = d.zmin, d.zmax
    return {
        "+X": [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        "-X": [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],
        "+Y": [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        "-Y": [(x0,y0,z0),(x0,y0,z1),(x1,y0,z1),(x1,y0,z0)],
        "+Z": [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        "-Z": [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0)],
    }

def disc_polygon(center, normal, radius, nverts=64):
    n = np.array(normal, dtype=float); n /= np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(np.dot(n,[1,0,0])) < 0.9 else np.array([0.0,1.0,0.0])
    u = a - np.dot(a, n)*n; u /= np.linalg.norm(u)
    v = np.cross(n, u)
    theta = np.linspace(0, 2*np.pi, nverts, endpoint=True)
    pts = [tuple((np.array(center) + radius*(np.cos(t)*u + np.sin(t)*v)).tolist()) for t in theta]
    return pts

def equalize_3d(ax):
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    s = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    cx, cy, cz = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim3d(cx - s/2, cx + s/2)
    ax.set_ylim3d(cy - s/2, cy + s/2)
    ax.set_zlim3d(cz - s/2, cz + s/2)

# ----------------------------
# 4) Plot Ω, Γₑ, Γ_out, Γ₀
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Ω: domain wireframe and faint array plane (z = 0)
faces = box_faces(D)
for fkey in faces:
    poly = Poly3DCollection([faces[fkey]], alpha=0.08, edgecolor="k", linewidths=0.6)
    ax.add_collection3d(poly)

plane_rect = [(D.xmin, D.ymin, D.array_plane_z),
              (D.xmax, D.ymin, D.array_plane_z),
              (D.xmax, D.ymax, D.array_plane_z),
              (D.xmin, D.ymax, D.array_plane_z)]
ax.add_collection3d(Poly3DCollection([plane_rect], alpha=0.05, facecolor="gray", edgecolor=None))

# Γₑ: 2×8 circular contacts on z=0 + current arrows (±z)
for e in ELECTRODES:
    verts = disc_polygon(center=e.center, normal=(0,0,1), radius=e.radius_mm, nverts=64)
    ax.add_collection3d(Poly3DCollection([verts], alpha=0.6))
    sign = 1.0 if e.current >= 0 else -1.0
    c = np.array(e.center, dtype=float)
    ax.quiver(c[0], c[1], c[2], 0, 0, sign, length=0.6, arrow_length_ratio=0.15, linewidth=1.2)
    ax.text(c[0], c[1], c[2], e.name, fontsize=7, ha="center", va="center")

# Γ₀: gauge/return disc on +Z face (Dirichlet φ=0)
g_verts = disc_polygon(center=GAUGE.center, normal=(0,0,1), radius=GAUGE.radius_mm, nverts=96)
ax.add_collection3d(Poly3DCollection([g_verts], alpha=0.25))
ax.text(GAUGE.center[0], GAUGE.center[1], GAUGE.center[2]+0.35, "Γ₀ (φ=0 V)", fontsize=9, ha="center")

# Γ_out: annotate six outer faces as insulating Neumann (j_n = 0)
face_centers = {
    "+X": (D.xmax, 0, 0), "-X": (D.xmin, 0, 0),
    "+Y": (0, D.ymax, 0), "-Y": (0, D.ymin, 0),
    "+Z": (0, 0, D.zmax), "-Z": (0, 0, D.zmin),
}
for k, ctr in face_centers.items():
    ax.text(ctr[0], ctr[1], ctr[2], f"{k}: Γ_out\n(j_n=0)", fontsize=7, ha="center", va="center")

# Labels, limits, aspect
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
ax.set_title("Tower A · 3D Geometry — 2×8 contacts at z=0, Γ₀ on +Z, Γ_out insulating")

ax.set_xlim(D.xmin, D.xmax)
ax.set_ylim(D.ymin, D.ymax)
ax.set_zlim(D.zmin, D.zmax)
equalize_3d(ax)
plt.tight_layout()
plt.show()

# ----------------------------
# 5) Export machine snapshot
# ----------------------------
contract = {
    "frame": "RAS",
    "units": {"length": "mm", "sigma": "S/m", "potential": "V"},
    "domain_mm": asdict(D),
    "electrodes": [
        {"name": e.name, "center_mm": e.center, "radius_mm": e.radius_mm,
         "bc": e.bc, "current": e.current}
        for e in ELECTRODES
    ],
    "gauge_patch": {
        "face": GAUGE.face, "center_mm": GAUGE.center,
        "radius_mm": GAUGE.radius_mm, "phi_V": GAUGE.phi_V
    },
    "outer_faces": {k: "neumann0" for k in ["+X","-X","+Y","-Y","+Z","-Z"]},
    "notes": "Positive arrow = +z (into tissue). Monopolar visual; Γ₀ acts as remote return."
}
with open("towerA_geometry_contract_2x8.json", "w") as f:
    json.dump(contract, f, indent=2)
print("Wrote: towerA_geometry_contract_2x8.json")
