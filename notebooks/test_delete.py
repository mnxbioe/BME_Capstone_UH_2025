# test_delete.py
import sys, os
# Add the src directory to Python's path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bme_capstone.tower_a import (
    Box3D, PlanePatch, TowerAGeometry,
    TowerALaplaceProblem, TowerABasisTrainer
)

# --------------------------------------------------------------------
# 1. Define geometry (a small 8x8x8 mm cube with one electrode patch)
# --------------------------------------------------------------------
volume = Box3D(x=(-4, 4), y=(-4, 4), z=(-4, 4))
contacts = [
    PlanePatch(
        name="E1",
        axis="z",
        value=0.0,
        span={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        normal_sign=1,
        kind="contact",
    ),
]
geometry = TowerAGeometry(volume=volume, contacts=contacts)

# --------------------------------------------------------------------
# 2. Define conductivity and stimulation current
# --------------------------------------------------------------------
sigma = 0.3  # S/m, typical gray matter conductivity
contact_currents = {"E2": -10e-6}  # 10 µA injected on E1
contact_currents = {"E1": 10e-6}  # 10 µA injected on E1
# --------------------------------------------------------------------
# 3. Build the Tower-A Laplace problem
# --------------------------------------------------------------------
trainer = TowerABasisTrainer(geometry=geometry, conductivity=sigma)
problem = trainer.build_problem(contact_currents=contact_currents)

# --------------------------------------------------------------------
# 4. Print a quick sanity check
# --------------------------------------------------------------------
print("Domains:", list(problem.domains.keys()))
print("Conditions:", list(problem.conditions.keys()))
print("Contact flux densities:", problem.contact_flux_densities)
print("✅ Tower-A Laplace problem built successfully!")
