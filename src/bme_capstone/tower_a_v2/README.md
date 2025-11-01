Tower A v2 (minimal, transparent)
================================

- Geometry lives in `geometry.py` as executable builders (start with `single_contact_reference()` for the one-electrode debug case).
- Boundary helpers with explicit units in `bc.py`.
- PINN problem/training scaffolding in `pinn_field.py` (ported from v1).
- Field evaluation helpers in `features.py`.
- Physics verification utilities in `verify/`.
- Run snapshot helper in `runlog.py` (creates `/runs/<timestamp>_<slug>_<sha>/`).
- Optional plotting utilities live in `visualize.py`; `a2_train_basis.py --plot` uses them (requires `matplotlib`).
- Dirichlet gauge is enforced via an ADF wrapper in `adf.py` (geometry.gauge specifies the patch).

Start scripts (`scripts/`):
- `a2_train_basis.py`
- `a2_verify.py`
- `a2_eval_grid.py`

`runs/` stays untracked (see `.gitignore`).
