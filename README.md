<<<<<<< HEAD
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
=======
Modeling Thalamocortical Responses to Microstimulation using Physics-Informed Neural Networks (PINN / CPIDINN)

University of Houston – Biomedical Engineering Capstone 2025



Problem



Microstimulation (MiSt) activates neurons, but predicting their responses is difficult because:



The neuron-to-electrode geometry is uncertain.



Electric fields from multiple electrodes overlap and interact in complex, non-additive ways.



Testing all possible electrode configurations experimentally is infeasible.



Proposed Solution



We build a two-tower physics-aware learning system that predicts cortical responses from thalamic stimulation patterns.



Tower A (Physics): Computes the electric potential and field in tissue from electrode currents using a Physics-Informed Neural Network (PINN / CPIDINN).



Tower B (Associative): Learns the mapping from field features to cortical RMS responses using supervised learning.



Together, these towers form a differentiable pipeline:



Currents (I) → Field (φ, E, J) → Response (ŷ)



This enables closed-loop optimization: adapting stimulation patterns based on predicted cortical outcomes.



Mathematical Foundation



The model enforces the quasi-static Laplace equation:



∇ · (σ ∇φ) = 0

E = −∇φ

J = σE



Boundary conditions:



Neumann (flux) on electrode surfaces for current-controlled injection



Dirichlet (ground) on distant boundaries for reference potential



Both enforced exactly through the Theory of Functional Connections (TFC)



The predicted response is computed from field magnitudes weighted by spatial sensitivity:



ŷ = σ(a \* Σ w(r) \* |σ(r) ∇φ(r)| + b)



Measured responses are RMS amplitudes of LFPs in a post-stimulus window.



Experimental Context



Input: Thalamic (VPL) electrode currents



Output: Cortical (S1) LFP RMS responses



Dataset: Derived from Dr. Joseph Francis’s associative MiSt experiments



Goal: Predict neural responses for new electrode configurations not seen during training.



Model Hierarchy





Tower A (PINN): Learned potential field satisfying Laplace PDE



Tower B (Supervised): Maps field features to RMS responses



CPIDINN Extension: Adds adaptive conductivity and equilibrium dynamics



Training and Evaluation



Stage 1: Freeze conductivity σ = σ₀, train potential and response heads with PDE + flux constraints



Stage 2: Unfreeze σ and add smoothness regularization to prevent overfitting



Optimizer: Adam (1e-3) with cosine learning-rate decay, optional LBFGS fine-tuning



Validation: Leave-One-Configuration-Out (LOCO) for generalization



Metric: R² per cortical channel using RMS responses in \[Ta, Tb]



Repository Structure

BME\_Capstone\_UH\_2025/

│

├── configs/            → YAML experiment settings

├── data/               → Raw \& processed data (gitignored)

├── notebooks/          → Jupyter notebooks

│   ├── 00\_setup.ipynb        ← environment initialization

│   ├── 01\_convert\_tdt.ipynb  ← data conversion (TDT)

│   ├── 02\_towerA\_train.ipynb ← physics-based field solver

│   └── 03\_towerB\_train.ipynb ← associative response model

│

├── scripts/            → CLI scripts for training \& evaluation

├── src/bme\_capstone/   → Core source code

│   ├── tower\_a/        ← PDE solvers, geometry, constraints

│   ├── tower\_b/        ← supervised mapping modules

│   └── utils/          ← metrics, seeding, and helpers

│

├── requirements.txt    → dependencies

└── README.md           → this file



Environment Setup

Local (Anaconda or JupyterLab)

git clone https://github.com/mnxbioe/BME\_Capstone\_UH\_2025.git

cd BME\_Capstone\_UH\_2025

pip install -r requirements.txt

jupyter lab



Google Colab

!git clone https://github.com/mnxbioe/BME\_Capstone\_UH\_2025.git

%cd BME\_Capstone\_UH\_2025

!pip install -r requirements\_colab.txt





Then open and run notebooks/00\_setup.ipynb.



References



Francis, J. T. \& Chapin, J. K. — Thalamocortical microstimulation response modeling (2012–2016)









Authors:

Melvin Nuñez, David De Lucio , Humberto 

Supervisors: Dr. Joseph Francis

University of Houston — Department of Biomedical Engineering

>>>>>>> 267d64a (Update all project files – configs, notebooks, scripts, and setup verification)
