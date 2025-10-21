from bme_capstone.tower_a import TowerALaplaceProblem, TowerABasisTrainer

trainer = TowerABasisTrainer(geometry=geometry, conductivity=sigma)
problem = trainer.build_problem(contact_currents=contact_currents)
print(problem.domains.keys())   # should list 'interior', 'contact:E1', etc.
print(problem.conditions.keys())  # PDE, contact flux, insulation, outer BCs
