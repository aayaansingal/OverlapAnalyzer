Overlap analysis module
===============================

This repository continues development of [OverlapAnalyzer](https://github.com/JunanLin/OverlapAnalyzer), whose underlying scientific software was developed mainly by Junan Lin. Its history up to `1f1510c` (2025-12-08) is taken directly from that repository.

### Differences from the original repository

New:
- **Parameter sweeps** (`overlapanalyzer/sweep`, `overlap-sweep` command): run many molecular configurations from one JSON config, with automatic target-state selection, cached and resumable results, and CSV summaries. See [Parameter sweeps](#parameter-sweeps).
- **Single-run script** (`scripts/run_overlap_simulation.py`) for trying one configuration by hand.
- **A PySCF-based Hamiltonian route** that works without the non-public `qcc_240202` package.
- **Automated tests** (`tests/`) and example configs (`examples/sweeps/`).

Fixed:
- `OverlapEstimator` crashed in `approx` and `below` modes (duplicate keyword arguments).
- `below` mode bounded the overlap *above* the energy threshold instead of below it (flipped energy rescaling).
- `approx` mode bounded the wrong region (or crashed) when the target level was not among the lowest two levels: target level indices were used as region indices after non-target levels were merged (`process_bound_pairs`).
- `calc_eigen`'s `s1` (open-shell singlet) reference was actually the Sz = 0 triplet (sign error in the two-determinant combination).
- `ham_gen` used RHF instead of ROHF for triplet active-space runs.
- Plotting crashed when no LaTeX installation was present.
- `krypy`, which fails to import on current SciPy, no longer breaks unrelated modules.
- `requirements.txt` now lists every package the code imports.

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
- Every result carries built-in checks. Two exact identities are verified: Hamiltonian moments versus spectrum (Σ overlap·Eᵏ = ⟨Hᵏ⟩), and the reference's ⟨S²⟩ versus its spin label. The run also notes when solver status is not `optimal` (e.g. SCS `optimal_inaccurate`), and whether each bound actually contains the exact value. Problems are listed in `warnings` (and in `results.csv`) without failing the experiment.
- A failed experiment is recorded with its error and the sweep continues. An interrupted sweep (Ctrl-C, also with `--jobs`) resumes from the first unfinished stage. If a parallel worker dies (e.g. out of memory), finished results are kept, the systems that were running are marked failed (`--retry-failed`), and systems that never started stay pending.
- Results are identified by a hash of their parameters, so changed parameters are recomputed and unchanged ones are reused. Each cached stage also records which code version and which upstream computation produced it. After a code fix, or a forced recompute (`--force`), affected results show as `stale` in `overlap-sweep status` and are recomputed on the next run.
- One `run` at a time per output directory: a lock file prevents two sweeps from writing the same results (exit code 3). With `--jobs N`, each worker gets an equal share of CPU threads.
- Outputs go to `sweeps/<sweep_name>/` by default (gitignored): `results.csv` (one row per experiment), `bounds.csv` (one row per mode and polynomial degree, with solver status), `results.jsonl`, and per-system folders with logs.
- `limits.max_dimension` (default 6000) refuses dense diagonalizations that would be too large; use an active space for bigger molecules.

Exit codes: 0 everything complete, 1 failures or unfinished work, 2 invalid config, 3 output directory in use, 130 interrupted.

Tests: `.venv/bin/python -m pytest tests/` (about 30 s). They check energies against Psi4 reference data shipped with OpenFermion and overlaps against PySCF FCI/CASCI wavefunctions. They also cover bound validity and monotonicity on random spectra, caching and invalidation, locking, crash recovery, parallel determinism, and interrupt/resume through the real CLI. `examples/sweeps/validation_suite.json` is a broader end-to-end check across molecules, spins, active spaces and bound modes.
