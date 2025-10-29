# Faster vectorized weighted-Jacobi Laplace solver with Neumann walls and Dirichlet electrode patches
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- geometry from paper ----------------------
row_spacing = 500e-6
col_spacing = 250e-6
n_rows, n_cols = 2, 8

A_tip = 1250e-12
r_eq = np.sqrt(A_tip / np.pi)

array_span_x = (n_cols - 1) * col_spacing
array_span_y = row_spacing
margin = 1.0e-3

Lx = array_span_x + 2*margin
Ly = array_span_y + 2*margin
Lz = 2.5e-3

# Smaller grid for speed (vectorized iterations)
nx = ny = nz = 81
dx, dy, dz = Lx/(nx-1), Ly/(ny-1), Lz/(nz-1)

x = np.linspace(-Lx/2,  Lx/2,  nx)
y = np.linspace(-Ly/2,  Ly/2,  ny)
z = np.linspace(0.0,    Lz,    nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

tip_z = Lz * 0.5
row_offsets = np.array([-row_spacing/2, +row_spacing/2])
col_offsets = (np.arange(n_cols) - (n_cols - 1)/2.0) * col_spacing
electrodes_xyz = np.array([(col_offsets[c], row_offsets[r], tip_z)
                           for r in range(n_rows) for c in range(n_cols)])

# Dirichlet sphere radius (inflate to at least ~2 cells)
min_cells = 2.0
r_numerical = max(r_eq, min_cells*max(dx, dy, dz))

def rc_to_index(r, c):
    return r*n_cols + c
pos_idx = rc_to_index(0, 3)
neg_idx = rc_to_index(0, 4)

V_pos, V_neg = +0.1, -0.1

V = np.zeros((nx, ny, nz), dtype=float)
dirichlet_mask = np.zeros_like(V, dtype=bool)
V_fix = np.zeros_like(V, dtype=float)

def add_dirichlet_sphere(center, radius, value):
    cx, cy, cz = center
    mask = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2 <= radius**2
    dirichlet_mask[mask] = True
    V_fix[mask] = value

add_dirichlet_sphere(electrodes_xyz[pos_idx], r_numerical, V_pos)
add_dirichlet_sphere(electrodes_xyz[neg_idx], r_numerical, V_neg)

# ---------------------- vectorized weighted-Jacobi ----------------------
inv_den = 1.0 / (2.0*(1/dx**2 + 1/dy**2 + 1/dz**2))
omega = 0.8                 # weighted-Jacobi factor (<=1)
max_iter = 1200
tol = 5e-6

for it in range(1, max_iter+1):
    # Neumann walls: mirror neighbors at boundaries
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

    # enforce Dirichlet
    V_new[dirichlet_mask] = V_fix[dirichlet_mask]

    # check convergence occasionally to reduce overhead
    if it % 20 == 0:
        err = np.max(np.abs(V_new - V))
        if err < tol:
            V = V_new
            break

    V = V_new

# ---------------------- fields & visualization ----------------------
Ex = -(np.roll(V,-1,axis=0) - np.roll(V,1,axis=0)) / (2*dx)
Ey = -(np.roll(V,-1,axis=1) - np.roll(V,1,axis=1)) / (2*dy)
Ez = -(np.roll(V,-1,axis=2) - np.roll(V,1,axis=2)) / (2*dz)
E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

k_slice = int(np.argmin(np.abs(z - tip_z)))
plt.figure(figsize=(7,6))
plt.imshow(E_mag[:,:,k_slice].T, origin='lower',
           extent=[x.min()*1e3, x.max()*1e3, y.min()*1e3, y.max()*1e3],
           aspect='equal')
plt.colorbar(label='|E| [V/m]')
plt.scatter([electrodes_xyz[pos_idx,0]*1e3, electrodes_xyz[neg_idx,0]*1e3],
            [electrodes_xyz[pos_idx,1]*1e3, electrodes_xyz[neg_idx,1]*1e3],
            s=30, marker='o')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title('Electric-field magnitude near electrode tips (mid-depth slice)')
plt.tight_layout()
plt.show()
