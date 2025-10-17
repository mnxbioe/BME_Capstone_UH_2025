# *Predicting Thalamocortical Responses to Microstimulation using Physics-Informed Neural Networks (PINN / CPIDINN)*

**University of Houston – Biomedical Engineering Capstone Team 6 2025**

---

##  Overview

This project models how electrical microstimulation (MiSt) in the thalamus (VPL) produces cortical responses in the somatosensory cortex (S1).
It integrates biophysics and data-driven learning into a two-tower system:

* **Tower A – Physics:**
  Solves the Laplace equation using a PINN to estimate tissue potentials and electric fields.
* **Tower B – Associative:**
  Learns the mapping from field features to measured neural responses through supervised training.

Together, these modules form an interpretable and differentiable chain:

```
Currents (I) → Electric Field (φ, E, J) → Cortical Response (ŷ)
```

---
## Training and Evaluation



Stage 1: Freeze conductivity σ = σ₀, train potential and response heads with PDE + flux constraints



Stage 2: Unfreeze σ and add smoothness regularization to prevent overfitting



Optimizer: Adam (1e-3) with cosine learning-rate decay, optional LBFGS fine-tuning



Validation: Leave-One-Configuration-Out (LOCO) for generalization



Metric: R² per cortical channel using RMS responses in \[Ta, Tb]

##  Key Features

* **Physics-constrained learning:** Enforces ∇·(σ∇φ)=0 with realistic boundary conditions
* **Field-aware modeling:** Uses features like |E| or |J| to predict cortical RMS activity
* **LOCO validation:** Tests generalization to new electrode configurations

---

##  Repository Layout

```
BME_Capstone_UH_2025/
├── notebooks/      → setup, data conversion, Tower A & B training
├── src/bme_capstone/
│   ├── tower_a/    → Laplace PDE / PINN models
│   ├── tower_b/    → associative mapping
│   └── utils/      → metrics, seeding, helpers
├── configs/        → YAML experiment settings
├── scripts/        → CLI utilities
├── requirements.txt
└── README.md
```

---

## Quick Start

**Local (Jupyter / Anaconda):**

```bash
git clone https://github.com/mnxbioe/BME_Capstone_UH_2025.git
cd BME_Capstone_UH_2025
pip install -r requirements.txt
jupyter lab
```

**Google Colab:**

```python
!git clone https://github.com/mnxbioe/BME_Capstone_UH_2025.git
%cd BME_Capstone_UH_2025
!pip install -r requirements_colab.txt
```

Then open and run `notebooks/00_setup.ipynb`.

---

##  References




Team Memebers: 
Melvin Nunez, David De Lucio , Humberto Acosta

Supervisors:
Dr. Joseph Francis

University of Houston — Department of Biomedical Engineering

