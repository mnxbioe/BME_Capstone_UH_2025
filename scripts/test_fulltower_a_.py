import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision("high")  # enables tensor core

 #---- EarlyStopping callback (added here) ----
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="pde_loss",      # <── matches the logged metric name
    min_delta=1e-12,
    patience=25,
    mode="min",
)

trainer_kwargs = {
    "max_epochs": 600,
    "accelerator": "gpu",
    "devices": 1,
    "enable_model_summary": False,
    "log_every_n_steps": 50,
    "callbacks": [early_stop],
}
# ---------------------------------------------
# path to src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bme_capstone.tower_a.geometry import Box3D, PlanePatch, TowerAGeometry
from bme_capstone.tower_a.pinn_field import (
    TowerALaplaceProblem, TowerABasisTrainer
)

# ------------------------------
# 1) Geometry (mm) & contacts
# ------------------------------
# 8 mm cube centered at origin
volume = Box3D(x=(-4, 4), y=(-4, 4), z=(-4, 4))

# Two 400x400 µm square contacts on z=0 plane, separated along x.
# (Span in mm: 0.4 mm square.)
csize = 0.2  # half-size (mm)
contacts = [
    PlanePatch(name="E_left",  axis="z", value=0.0, span={"x": (-0.8-csize, -0.8+csize), "y": (-csize, csize)}, normal_sign=+1, kind="contact"),
    PlanePatch(name="E_right", axis="z", value=0.0, span={"x": (+0.8-csize, +0.8+csize), "y": (-csize, csize)}, normal_sign=+1, kind="contact"),
]

# Optional: an outer boundary at z=+/-4 mm (top/bottom planes) as Dirichlet φ=0 to anchor potential.
# (You can add more 'outer' planes if you’d like; this is sufficient for a quick test.)
outers = [
    PlanePatch(name="z_top",    axis="z", value=+4.0, span={"x": (-4, 4), "y": (-4, 4)}, normal_sign=+1, kind="outer"),
    PlanePatch(name="z_bottom", axis="z", value=-4.0, span={"x": (-4, 4), "y": (-4, 4)}, normal_sign=-1, kind="outer"),
]

geometry = TowerAGeometry(volume=volume, contacts=contacts, shanks=[], outers=outers)

# ------------------------------
# 2) Conductivity & currents
# ------------------------------
# Use mm units consistently. σ=0.3 S/m -> 0.0003 S/mm
sigma_SI = 0.3  # S/m (gray matter typical)
sigma = sigma_SI / 1000.0  # S/mm

# currents in Amps
I = 10e-6  # 10 µA

# Three patterns: left-only, right-only, and both (superposition test)
patterns = {
    "left_only":  {"E_left": I,  "E_right": 0.0},
    "right_only": {"E_left": 0.0,"E_right": -I},
    "both":       {"E_left": +I,  "E_right": -I},
}

# Dirichlet outer BC (φ=0) on the top/bottom planes
outer_bc = {
    "z_top":    {"type": "dirichlet", "value": 0.0},
    "z_bottom": {"type": "dirichlet", "value": 0.0},
}

# ------------------------------
# 3) Train PINN for each pattern
# ------------------------------
trainer = TowerABasisTrainer(
    geometry=geometry,
    conductivity=sigma,         # scalar is fine
    # You can swap in RBAPINN/SelfAdaptivePINN once everything runs:
    # solver_cls=RBAPINN
)

def train_pattern(name, currents):
    res = trainer.train_basis(
        contact_name="dummy",  # ignored; we pass a full pattern below
        current=I,             # ignored
        outer_bc=outer_bc,
        solver_kwargs={"use_lt": True},
        trainer_kwargs={"max_epochs": 600, "accelerator": "gpu", "enable_model_summary": False, "log_every_n_steps": 50},
        discretisation_kwargs={
            "interior_points": 40000,
            "contact_points":  6000,
            "outer_points":    4000,
            "interior_mode":   "latin",
            "boundary_mode":   "random",
        },
        model=None,  # default MLP 6x128 tanh
    )
    return res

# Small change: let’s override the problem construction to use the full current dict
def build_and_train(currents):
    problem = trainer.build_problem(contact_currents=currents, outer_bc=outer_bc)
    trainer.discretise(problem,
                       interior_points=40000,
                       contact_points=6000,
                       outer_points=4000,
                       interior_mode="latin",
                       boundary_mode="random")
    model = trainer.build_model()
    solver = trainer.solver_cls(problem=problem, model=model, use_lt=True)
    tr = trainer.Trainer(
        solver=solver,
        train_size=1.0, val_size=0.0, test_size=0.0, batch_size=None,
        **trainer_kwargs  # <── injects all the settings above
    )

    tr.train()
    return problem, model, solver, tr

results = {}
for key, cc in patterns.items():
    problem, model, solver, tr = build_and_train(cc)
    results[key] = {"problem": problem, "model": model, "solver": solver}

# ------------------------------
# 4) Sanity checks
# ------------------------------
def eval_phi(model, xs, ys, zs):
    # xs, ys, zs are 1D arrays (mm)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        out = model(pts).cpu().numpy()  # output variable is ['phi']
    return X, Y, Z, out.reshape(X.shape)

# (a) PDE residual magnitude (sample a few random interior points)
from pina import LabelTensor
from pina.operator import grad, div

def pde_residual_mean(problem, model, n=5000):
    """Compute mean absolute PDE residual |∇·(σ∇φ)| over random points."""
    pts_sample = problem.spatial_domain.sample(n=n, mode="random")
    pts_tensor = pts_sample.tensor if isinstance(pts_sample, LabelTensor) else pts_sample
    pts = LabelTensor(pts_tensor, labels=["x", "y", "z"])
    pts.requires_grad_(True)

    out_raw = model(pts)
    out_tensor = out_raw.tensor if isinstance(out_raw, LabelTensor) else out_raw
    out = LabelTensor(out_tensor, labels=["phi"])

    g = grad(out, pts, components=["phi"], d=["x","y","z"])
    sigma_val = float(sigma)
    flux = sigma_val * g.tensor
    flux_lt = LabelTensor(flux, labels=["J_x", "J_y", "J_z"])
    res = div(flux_lt, pts, components=flux_lt.labels, d=["x","y","z"]).tensor
    return float(res.abs().mean().cpu())



# (b) Net flux balance on contacts (sum target flux vs. model gradient)
#    (Quick proxy: just print the per-contact target flux densities you set.)
for key in patterns:
    print(f"\nPattern {key} target flux densities (A/mm^2):")
    prob = results[key]["problem"]
    print(f"\nPattern {key} contact flux densities (A/mm²):")
    for k, s in prob.surface_conditions.items():
        if s.condition_type == "neumann" and "contact" in k:
            print(f"  {s.surface.name:10s} → {s.description}")


# (c) Superposition: φ_left + φ_right ≈ φ_both
xs = np.linspace(-2.0, 2.0, 121)
ys = np.linspace(-2.0, 2.0, 121)
zs = np.array([0.5])   # one slice above the plane

X, Y, Z, phi_left  = eval_phi(results["left_only"]["model"],  xs, ys, zs)
_,  _,  _, phi_right = eval_phi(results["right_only"]["model"], xs, ys, zs)
_,  _,  _, phi_both  = eval_phi(results["both"]["model"],      xs, ys, zs)

lin_err = np.mean(np.abs((phi_left + phi_right) - phi_both)) / np.maximum(1e-9, np.mean(np.abs(phi_both)))
print(f"\nSuperposition relative error on slice z=0.5 mm: {lin_err:.3e}")

# PDE residuals
for key in patterns:
    rm = pde_residual_mean(results[key]["problem"], results[key]["model"], n=4000)
    print(f"PDE residual mean |∇·(σ∇φ)| for {key}: {rm:.3e}")

# ------------------------------
# 5) Quick plots
# ------------------------------
import matplotlib
matplotlib.use("TkAgg")  # or 'QtAgg' if you have PyQt

fig, axs = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True)
for ax, dat, ttl in zip(
    axs,
    [phi_left[:, :, 0], phi_right[:, :, 0], phi_both[:, :, 0]],
    ["φ (left only)", "φ (right only)", "φ (both)"],
):
    im = ax.contourf(X[:, :, 0], Y[:, :, 0], dat, levels=20)
    ax.set_title(ttl); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)

# 1-D cut along x at y=0,z=0.5
y0_idx = np.argmin(np.abs(ys - 0.0))
plt.figure(figsize=(6,4))
plt.plot(xs, phi_left[:, y0_idx, 0],  label="left")
plt.plot(xs, phi_right[:, y0_idx, 0], label="right")
plt.plot(xs, phi_both[:, y0_idx, 0],  label="both")
plt.title("Line cut (y=0, z=0.5 mm)")
plt.xlabel("x [mm]"); plt.ylabel("φ")
plt.legend(); plt.tight_layout()
plt.show()
