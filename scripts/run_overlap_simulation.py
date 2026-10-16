"""
Single overlap simulation: molecule -> Hamiltonian -> exact eigen data -> polynomial overlap bounds.

Edit the CONFIGURATION section below, then run from the repository root:

    .venv/bin/python scripts/run_overlap_simulation.py

This is the hands-on, one-system version of the sweep framework (overlapanalyzer.sweep), and it reuses
the sweep's Hamiltonian and diagonalization stages. To scan many geometries, bases or spins with cached,
resumable results, write a sweep config instead (see examples/sweeps/) and run `overlap-sweep run CONFIG`.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # Save plots to files instead of opening windows

from overlapanalyzer.polynomial_majorisation_LB import OverlapEstimator
from overlapanalyzer.read_ham import ensure_directory_exists
from overlapanalyzer.sweep.config import SystemSpec
from overlapanalyzer.sweep.hamiltonian import generate_hamiltonian
from overlapanalyzer.sweep.spectrum import run_exact_diagonalization


# ============================== CONFIGURATION ==============================

# --- Molecule (any molecule from the sweep config format works here too) ---
system = SystemSpec(
    molecule="h_chain",         # 'h_chain', or one of h2, lih, beh2, h2o, n2, h4, h4_linear, nh3, h2x2
    n_atoms=4,                  # h_chain only
    geometry=1.5,               # Bond length in Angstrom for chains and most library molecules
    basis="sto-3g",
    spin="s0",                  # 's0' (singlet HF reference), 't1' or 's1' (see eigen.calc_eigen)
    active_space=None,          # e.g. {"active_electrons": 2, "active_spatial_orbitals": 5}
)

# --- Overlap-bound recipe (passed to OverlapEstimator.compute_bounds) ---
# target_indices refer to distinct energy levels sorted from lowest energy (0 = lowest level).
recipe = {
    "mode": ["exact", "approx"],    # Any of "exact", "approx", "below", "manual"
    "poly_degrees": [2, 4, 6, 8],   # Polynomial degrees = number of Hamiltonian moments used
    "target_indices": [0],          # Which eigenstates to bound the overlap with
    "b_values": [1],                # Weight of each target state in the overlap being bounded
    "bound_extension": 0.05,        # approx mode: eigenvalue uncertainty as a fraction of the gap to each neighbour (< 0.5)
    "solver": "SCS",                # approx/below: "SCS" (free) or "MOSEK" (needs a license)
}

# --- Output ---
output_root = "gen_hams"        # Gitignored; each system gets its own subfolder
save_plots = True
max_dimension = 6000            # Refuse dense diagonalizations larger than this

# ===========================================================================


def print_bounds(results):
    payloads = results if isinstance(results, list) else [results]
    for payload in payloads:
        exact_overlap = payload.get("exact_overlap")
        print(f"\nMode: {payload['mode']}" + (f"   (exact overlap = {exact_overlap:.6f})" if exact_overlap is not None else ""))
        print(f"{'degree':>8} {'lower':>12} {'upper':>12}")
        for entry in payload["degree_results"]:
            print(f"{entry['degree']:>8} {entry['lower']:>12.6f} {entry['upper']:>12.6f}")


def main():
    data_dir = os.path.join(output_root, system.label)
    ensure_directory_exists(data_dir)

    # Stage 1: Hamiltonian
    info = generate_hamiltonian(system, data_dir)
    print(f"{system.label}: {info.n_electrons} electrons, {info.n_spin_orbitals} spin orbitals, "
          f"number-conserving dimension {info.hilbert_dimension}, HF energy {info.hf_energy:.8f}")

    # Stage 2: exact spectrum, overlaps of the reference state, and Hamiltonian moments
    eigen_path = run_exact_diagonalization(system, info, data_dir, max_dimension)
    with open(eigen_path, "r") as eigen_file:
        eig_results = json.load(eigen_file)
    print(f"Lowest distinct energies: {eig_results['exact_energies_truncated'][:5]}")
    print(f"Their exact overlaps:     {eig_results['overlaps_truncated'][:5]}")

    # Stage 3: polynomial bounds on the overlap from the moments
    results_dir = os.path.join(data_dir, "bounds")
    ensure_directory_exists(results_dir)
    estimator = OverlapEstimator()
    results = estimator.compute_bounds({**recipe, "eig_results": eig_results},
                                       visualize=save_plots, save_plots=save_plots, save_dir=results_dir)
    print_bounds(results)
    print(f"\nResults written to {results_dir}")


if __name__ == "__main__":
    main()
