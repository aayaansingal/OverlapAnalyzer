Overlap analysis module
===============================

This repository continues development of [OverlapAnalyzer](https://github.com/JunanLin/OverlapAnalyzer), whose underlying scientific software was developed mainly by Junan Lin. Its history up to `1f1510c` (2025-12-08) is taken directly from that repository.

Library install
---------------

After downloading the repository, it can be installed as a Python library with:

```
  pip install -r requirements.txt
  pip install -e .
```
The dependency on QuantumMAMBO needs some extra care since the juliacall package will by default install the latest version of Julia in your environment, which will not be compatible with QuantumMAMBO. A temporary resolution is as follows (assuming the virtual environment is located at /.venv):

(1) Locate the .venv/julia_env/pyjuliapkg folder

(2) Copy the juliapkg.json file to that folder

(3) Run
```
python setup-julia.py
```

Some dependencies need separate handling:

- `qcc_240202` (iQCC/QCC routines used by `ham_gen`, `ham_conversion` and `iQCC`) is not on PyPI and must be installed separately.
- `Mosek` (used by `polynomial_majorisation_LB` through `cvxpy`) needs a MOSEK license file.
- `krypy` is only needed for `alg_LinearOp.calculateInnerProducts` and `precond_utils`. Its latest release (2.2.0) does not import on recent SciPy versions.

Usage
-----
Calculating wavefunction overlaps with Hamiltonian eigenstates from Hamiltonian power expectation values.
arXiv: https://arxiv.org/abs/2503.12224

Running a simulation
--------------------
`scripts/run_overlap_simulation.py` runs the full pipeline: build a molecular Hamiltonian, compute the exact spectrum with `eigen.calc_eigen`, then bound the reference state's overlap with `OverlapEstimator.compute_bounds`. Edit the CONFIGURATION block at the top of the script, then run from the repository root:

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python scripts/run_overlap_simulation.py
```

Output goes to `gen_hams/<molecule>/`, which is gitignored: `ham_fer/` (Hamiltonian), `eigen/` (spectrum, overlaps and moments as JSON) and `bounds/` (`all_payloads.json` plus per-degree plots).

Parameter sweeps
----------------
`overlapanalyzer.sweep` automates scanning many molecular configurations. A JSON config lists systems (molecule, geometries, basis sets, reference spins `s0`/`t1`/`s1`, optional frozen-core active spaces) and analyses (how to choose the target eigenstate and which overlap bounds to compute). Every system x analysis combination is one experiment. Examples are in `examples/sweeps/`.

```
overlap-sweep validate examples/sweeps/h_chain_bond_scan.json   # check the config, list experiments
overlap-sweep run examples/sweeps/h_chain_bond_scan.json        # run; rerun the same command to resume
overlap-sweep run CONFIG --jobs 4                               # process systems in parallel
overlap-sweep run CONFIG --retry-failed                         # recompute experiments that failed
overlap-sweep status CONFIG                                     # complete / failed / pending per experiment
```
(`python -m overlapanalyzer.sweep ...` works the same way without reinstalling.)

How it works:

- Each system runs Hamiltonian generation (backend `pyscf` via openfermionpyscf, or `ham_gen` when `qcc_240202` is installed), then `eigen.calc_eigen`. The results are cached per system, so adding or changing analyses never repeats a diagonalization.
- Target selection is automatic: `lowest_with_overlap` (default), `max_overlap`, `index`, or `lowest_in_sector`, which picks the lowest level whose spin matches the reference, even if the true ground state is in another spin sector.
- Bounds come from `OverlapEstimator` in `exact`, `approx` and `below` modes. Each mode runs separately, so one solver failure does not lose the others.
- `result.json` keeps three things separate: **exact** quantities from diagonalization (energies, overlaps, spin, gaps), **bounds** (estimates from Hamiltonian moments, each stored next to the exact value it bounds plus a `brackets_exact` check), and timings.
- A failed experiment is recorded with its error and the sweep continues. An interrupted sweep (Ctrl-C) resumes from the first unfinished stage. Results are identified by a hash of their parameters, so changed parameters are recomputed and unchanged ones are reused.
- Outputs go to `sweeps/<sweep_name>/` by default (gitignored): `results.csv` (one row per experiment), `bounds.csv` (one row per mode and polynomial degree), `results.jsonl`, and per-system folders with logs.
- `limits.max_dimension` (default 6000) refuses dense diagonalizations that would be too large; use an active space for bigger molecules.

Tests: `.venv/bin/python -m pytest tests/`.
