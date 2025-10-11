# *Predicting Thalamocortical Responses to Microstimulation using Physics-Informed Neural Networks (PINN / CPIDINN)*

**University of Houston – Biomedical Engineering Capstone 2025**

---

## 📘 Overview

This project models how **electrical microstimulation (MiSt) in the thalamus (VPL)** produces **cortical responses in the somatosensory cortex (S1)**.
It integrates biophysics and data-driven learning into a **two-tower system**:

* **Tower A – Physics:**
  Solves the **Laplace equation** using a **Physics-Informed Neural Network (PINN)** to estimate tissue potentials and electric fields.
* **Tower B – Associative:**
  Learns the mapping from field features to measured neural responses (RMS amplitudes) through supervised training.

Together, these modules form an interpretable and differentiable chain:

```
Currents (I) → Electric Field (φ, E, J) → Cortical Response (ŷ)
```

---

## ⚙️ Key Features

* **Physics-constrained learning:** Enforces ∇·(σ∇φ)=0 with realistic boundary conditions
* **Field-aware modeling:** Uses features like |E| or |J| to predict cortical RMS activity
* **LOCO validation:** Tests generalization to new electrode configurations
* **Unified workflow:** Works in both **JupyterLab (local)** and **Google Colab**

---

## 🧩 Repository Layout

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

## 🚀 Quick Start

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

## 📚 References

* **Francis & Chapin (2012–2016):** Associative thalamocortical microstimulation
* **Raissi et al. (2019):** Physics-Informed Neural Networks (J. Comput. Phys.)
* **Psaros et al. (2023):** Hard-Constrained PINNs via Theory of Functional Connections
