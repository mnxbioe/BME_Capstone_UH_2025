import sys
from pathlib import Path
import torch

# Make sure Python can find your folders
ROOT = Path.cwd()
SRC = ROOT / "src"
for p in [ROOT, SRC]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Import what we need
from bme_capstone.tower_a import TowerABasisTrainer
from bme_capstone.tower_a.geometry import TowerAGeometry
from scripts.Tower_A_TestV1.one_electrode_train_and_viz import _plot_slices
torch.serialization.add_safe_globals([TowerAGeometry])

# Reload saved geometry and model
geometry = torch.load(
    r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\scripts\Tower_A_TestV1\geometry_obj.pt",
    weights_only=False
)
trainer = TowerABasisTrainer(geometry=geometry, conductivity=0.3/1000.0)
model = trainer.build_model()
model.load_state_dict(torch.load(
    r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\scripts\Tower_A_TestV1\towerA_single_electrode.pt"
))
model.eval()

print("✅ Model and _plot_slices imported successfully!")
if __name__ == "__main__":
    _plot_slices(model, geometry)
from pina import LabelTensor
import numpy as np

def sample_field(model, n=2000):
    """Sample random points in the domain and compute ∇ϕ."""
    xs = torch.empty((n, 3)).uniform_(-2.0, 2.0)
    pts = LabelTensor(xs, labels=["x", "y", "z"])
    pts.requires_grad_(True)
    out = model(pts)
    phi = out.tensor if isinstance(out, LabelTensor) else out
    grads = torch.autograd.grad(
        phi.sum(), pts.tensor, create_graph=False, retain_graph=False
    )[0]
    E = -grads
    return phi.detach(), E.detach()

phi, E = sample_field(model)
print("Mean potential:", phi.mean().item(), "V (arbitrary)")
print("Mean |E|:", E.norm(dim=1).mean().item(), "A.U.")
import torch
from pathlib import Path
from bme_capstone.tower_a.geometry import TowerAGeometry
torch.serialization.add_safe_globals([TowerAGeometry])

ROOT = Path("C:/Users/Melvi/02.1_Coding_projects/BME_Capstone_UH_2025/BME_Capstone_UH_2025_Github/scripts/Tower_A_TestV1")

# --- Geometry inspection ---
geom_path = ROOT / "geometry_obj.pt"
geometry = torch.load(geom_path, weights_only=False)

print("\n=== Geometry summary ===")
print("Interior volume (mm):")
print("  x:", geometry.volume.x)
print("  y:", geometry.volume.y)
print("  z:", geometry.volume.z)
print("Contacts:", [p.name for p in geometry.contacts])
print("Outer walls:", [p.name for p in geometry.outers])
print("Patch types:", [p.kind for p in geometry.contacts + geometry.outers])

# --- Model inspection ---
model_path = ROOT / "towerA_single_electrode.pt"
state = torch.load(model_path, map_location="cpu")
print("\n=== Model checkpoint summary ===")
print("Keys in state dict:", list(state.keys())[:5], "...")
print("Total parameters:", sum(p.numel() for p in state.values()))
