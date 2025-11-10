import json, csv, re
import numpy as np
import matplotlib.pyplot as plt

# -------------------- helpers (from the files we generated) --------------------
def load_geom(path="vpl_array_geometry.json"):
    with open(path, "r") as f:
        g = json.load(f)
    pts = [(int(e["index"]), float(e["x"]), float(e["y"]), float(e["z"])) for e in g["electrodes"]]
    A = float(g["tip"]["area_m2"]); r = float(g["tip"]["radius_m"])
    return pts, A, r, g

def active_electrode_signs_at(t_s, pulse_csv="pulse_schedule.csv"):
    """
    Return dict: {electrode_index: +1 | -1} for electrodes that are ON at time t_s,
    based on the 'note' field in the CSV (phase1_pos(+I), phase1_neg(-I), ...).
    """
    sig = {}
    with open(pulse_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            t0 = float(row["t_start_s"]); t1 = float(row["t_end_s"])
            if not (t0 <= t_s < t1): continue
            ei = int(row["electrode_index"])
            note = row.get("note","")
            # If current is +I, treat that electrode as the "positive" terminal, else negative.
            sgn = +1 if "+I" in note else -1
            # If multiple overlapping segments ever exist, last one wins (not typical).
            sig[ei] = sgn
    return sig

# -------------------- load geometry & choose a sample time --------------------
pts, A_tip, r_tip, geom = load_geom("vpl_array_geometry.json")
print(f"Loaded {len(pts)} electrodes; tip radius ~ {r_tip*1e6:.1f} µm  (area {A_tip*1e12:.0f} µm²)")

# Pick a time during the first burst (e.g., 10 ms)
t_sample = 0.0061  # seconds
signs = active_electrode_signs_at(t_sample, "pulse_schedule.csv")
if not signs:
    raise SystemExit("At t_sample, no pulses are active. Pick a different time (e.g., 0.010 or 0.175 s).")
print("Active electrodes at t = %.2f ms:" % (t_sample*1e3), signs)

# -------------------- 3D grid domain --------------------
# Use the domain hinted in the geometry file, or choose your own.
Lx = float(geom["domain_hint"]["Lx_m"]) if "domain_hint" in geom else 3.0e-3
Ly = float(geom["domain_hint"]["Ly_m"]) if "domain_hint" in geom else 2.0e-3
Lz = float(geom["domain_hint"]["Lz_m"]) if "domain_hint" in geom else 2.5e-3

nx = ny = nz = 91
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
z = np.linspace(0.0, Lz, nz)  # depth
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# -------------------- Dirichlet electrode patches from active signs --------------------
# Inflate radius to span at least ~2 grid cells for stability
inflate = 2.0 * max(dx, dy, dz)
r_num = max(r_tip, inflate)

V0 = 0.1  # Volts amplitude for visualization (Dirichlet). Linear problem => scale to taste.
V = np.zeros((nx, ny, nz), dtype=float)
dirichlet_mask = np.zeros_like(V, dtype=bool)
V_fix = np.zeros_like(V, dtype=float)

for (ei, sx, sy, sz) in pts:
    if ei not in signs: 
        continue  # only patch the ones ON right now
    sgn = signs[ei]
    mask = ((X-sx)**2 + (Y-sy)**2 + (Z-sz)**2) <= r_num**2
    dirichlet_mask |= mask
    V_fix[mask] = sgn * V0

n_active_voxels = int(dirichlet_mask.sum())
print(f"Patched {len(signs)} active electrodes; ~{n_active_voxels} voxels fixed.")

# -------------------- Solve Laplace with insulating outer walls (Neumann on box) --------------------
# Vectorized weighted-Jacobi (fast & simple). Neumann walls by mirroring neighbors.
inv_den = 1.0 / (2.0*(1/dx**2 + 1/dy**2 + 1/dz**2))
omega = 0.8
max_iter = 2000
tol = 5e-6

for it in range(1, max_iter+1):
    Vxm = np.empty_like(V); Vxp = np.empty_like(V)
    Vym = np.empty_like(V); Vyp = np.empty_like(V)
    Vzm = np.empty_like(V); Vzp = np.empty_like(V)

    Vxm[1:,:,:] = V[:-1,:,:];   Vxm[0,:,:]  = V[1,:,:]
    Vxp[:-1,:,:] = V[1:,:,:];   Vxp[-1,:,:] = V[-2,:,:]

    Vym[:,1:,:] = V[:,:-1,:];   Vym[:,0,:]  = V[:,1,:]
    Vyp[:,:-1,:] = V[:,1:,:];   Vyp[:,-1,:] = V[:,-2,:]

    Vzm[:,:,1:] = V[:,:,:-1];   Vzm[:,:,0]  = V[:,:,1]
    Vzp[:,:,:-1] = V[:,:,1:];   Vzp[:,:,-1] = V[:,:,-2]

    rhs = (Vxm + Vxp)/dx**2 + (Vym + Vyp)/dy**2 + (Vzm + Vzp)/dz**2
    V_new = (1-omega)*V + omega*rhs*inv_den

    # enforce Dirichlet on electrode patches
    V_new[dirichlet_mask] = V_fix[dirichlet_mask]

    if it % 20 == 0:
        err = np.max(np.abs(V_new - V))
        # print(f"iter {it} err={err:.2e}")
        if err < tol:
            V = V_new
            break
    V = V_new

# Remove gauge drift (optional): zero-mean
V -= V.mean()

# -------------------- Fields & plots --------------------
Ex = -(np.roll(V,-1,axis=0) - np.roll(V,1,axis=0)) / (2*dx)
Ey = -(np.roll(V,-1,axis=1) - np.roll(V,1,axis=1)) / (2*dy)
Ez = -(np.roll(V,-1,axis=2) - np.roll(V,1,axis=2)) / (2*dz)
E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

# Slice at the tip plane (closest z to the tips we stored)
tip_z = float(geom["tip_plane_z_m"])
k_slice = int(np.argmin(np.abs(z - tip_z)))

plt.figure(figsize=(7,6))
plt.imshow(E_mag[:,:,k_slice].T, origin='lower',
           extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3],
           aspect='equal')
plt.colorbar(label='|E| [V/m]')
# scatter the active electrode centers
act = [(x0,y0) for (ei,x0,y0,z0) in pts if ei in signs]
if act:
    xs, ys = np.array(act).T
    plt.scatter(xs*1e3, ys*1e3, s=20, c='k')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f'|E| slice at z≈tip (t={t_sample*1e3:.1f} ms)')
plt.tight_layout()
plt.show()

# Save for reuse
np.savez("field_snapshot.npz", x=x, y=y, z=z, V=V, Ex=Ex, Ey=Ey, Ez=Ez, E_mag=E_mag,
         t_sample=t_sample, active_signs=signs, r_num=r_num, V0=V0)
print("Saved field_snapshot.npz")
