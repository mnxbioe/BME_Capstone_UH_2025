# Tower A v2 Layout (Physics-First)

This mirrors the QA plan so contributors know exactly where to look.

- `docs/` – living docs (this layout note, QA excerpts, future baselines).
- `src/bme_capstone/tower_a_v2/`
  - `geometry.py` – geometry primitives + named builders (no JSON). Current starters: `single_contact_reference()` for single-electrode debugging and `example_two_contacts()` for future multi-contact tests.
  - `bc.py` – boundary helpers with explicit units (A/mm^2, S/mm).
  - `pinn_field.py` – Laplace problem, trainers, constants, verification hooks.
  - `features.py` – grid generation, field evaluation, feature extraction.
  - `adf.py` – distance utilities + wrapper enforcing the Dirichlet gauge via ADF.
  - `verify/` – physics checks (superposition, net-current, residuals, health).
  - `runlog.py` – run folder + config snapshot utilities.
- `scripts/`
  - `a2_train_basis.py` – train one basis (explicit geometry + preset).
    - pass `--plot` to visualise |E| slices and collocation samples (requires matplotlib)
  - `a2_verify.py` – fast physics suite for CI/operators.
  - `a2_eval_grid.py` – interpretability/grid exports (stub until models ported).
- `tests/`
  - (TODO) CPU smoke + superposition tests (<2 min) matching QA doc.
- `runs/` – untracked per-run outputs (config snapshots, reports, health).

Tower A v1 remains for historical parity; v2 focuses on physics checks and
clean traceability. All new work should import `bme_capstone.tower_a_v2.*`.
