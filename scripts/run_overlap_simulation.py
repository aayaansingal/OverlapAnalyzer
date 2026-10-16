"""
End-to-end overlap simulation: molecule -> Hamiltonian -> exact eigen data -> polynomial overlap bounds.

Edit the CONFIGURATION section below, then run from the repository root:

    .venv/bin/python scripts/run_overlap_simulation.py

Stage 1 builds the fermionic Hamiltonian with openfermionpyscf (a stand-in for ham_gen.gen_hamiltonian,
which needs the non-public qcc_240202 package) and saves it in the same file layout ham_gen uses.
Stages 2 and 3 use the library directly: eigen.calc_eigen and polynomial_majorisation_LB.OverlapEstimator.
"""
import os
import csv
import json
from math import comb

import matplotlib
matplotlib.use("Agg")  # Save plots to files instead of opening windows

from openfermion import MolecularData, get_fermion_operator, save_operator
from openfermionpyscf import run_pyscf

from overlapanalyzer.eigen import calc_eigen
from overlapanalyzer.polynomial_majorisation_LB import OverlapEstimator
from overlapanalyzer.read_ham import ensure_directory_exists


# ============================== CONFIGURATION ==============================

# --- Molecule ---
mol_name = "H4"                 # Label used in output filenames
num_hydrogen = 4                # Linear hydrogen chain with this many atoms
bond_length = 1.5               # Angstrom, spacing between neighbouring atoms
basis = "sto-3g"
spin_label = "s0"               # 's0' (singlet HF reference), 't1' or 's1' (see eigen.calc_eigen)

# --- Overlap-bound recipe (passed to OverlapEstimator.compute_bounds) ---
# Indices refer to the distinct eigenvalues (with nonzero overlap) sorted from lowest energy.
recipe = {
    "mode": ["exact", "approx"],    # Any of "exact", "approx", "below", "manual"
    "poly_degrees": [2, 4, 6, 8],   # Polynomial degrees = number of Hamiltonian moments used
    "target_indices": [0],          # Which eigenstates to bound the overlap with (0 = ground state)
    "b_values": [1],                # Weight of each target state in the overlap being bounded
    "bound_extension": 0.05,        # approx mode: half-width of the uncertainty window around each eigenvalue
    "solver": "SCS",                # approx/below: "SCS" (free) or "MOSEK" (needs a license)
}

# --- Output ---
output_root = "gen_hams"        # Gitignored; each molecule gets its own subfolder
save_plots = True

# ===========================================================================


def build_hydrogen_chain(num_atoms, spacing):
    return [("H", (0.0, 0.0, i * spacing)) for i in range(num_atoms)]


def generate_hamiltonian(data_dir, geometry_label):
    """Run RHF with PySCF and save the fermionic Hamiltonian where calc_eigen expects it."""
    geometry = build_hydrogen_chain(num_hydrogen, bond_length)
    multiplicity = 3 if spin_label == "t1" else 1
    molecule = MolecularData(geometry, basis, multiplicity, charge=0,
                             filename=os.path.join(data_dir, f"{mol_name}_{geometry_label}"))
    molecule = run_pyscf(molecule, run_scf=True)

    n_elec = molecule.n_electrons
    n_spin_orb = 2 * molecule.n_orbitals
    hilbert_dim = comb(n_spin_orb, n_elec)
    print(f"{mol_name}: {n_elec} electrons, {n_spin_orb} spin orbitals, "
          f"number-conserving dimension {hilbert_dim}, HF energy {molecule.hf_energy:.8f}")
    if hilbert_dim > 5000:
        print("Warning: calc_eigen diagonalizes a dense matrix of this dimension; this may be slow.")

    hamiltonian_fermion = get_fermion_operator(molecule.get_molecular_hamiltonian())
    ham_dir = os.path.join(data_dir, "ham_fer")
    ensure_directory_exists(ham_dir)
    ham_filename = f"{mol_name}_{geometry_label}_{n_elec}_{n_spin_orb}_{spin_label}.data"
    save_operator(hamiltonian_fermion, file_name=ham_filename, data_directory=ham_dir,
                  allow_overwrite=True, plain_text=True)

    # Same info.csv layout as ham_gen.gen_hamiltonian
    with open(os.path.join(data_dir, "info.csv"), "w") as info_file:
        writer = csv.writer(info_file)
        writer.writerow(["N_electron", "N_spin_orb"])
        writer.writerow([n_elec, n_spin_orb])

    return molecule.hf_energy, n_elec, n_spin_orb


def print_bounds(results):
    payloads = results if isinstance(results, list) else [results]
    for payload in payloads:
        exact_overlap = payload.get("exact_overlap")
        print(f"\nMode: {payload['mode']}" + (f"   (exact overlap = {exact_overlap:.6f})" if exact_overlap is not None else ""))
        print(f"{'degree':>8} {'lower':>12} {'upper':>12}")
        for entry in payload["degree_results"]:
            print(f"{entry['degree']:>8} {entry['lower']:>12.6f} {entry['upper']:>12.6f}")


def main():
    geometry_label = f"{bond_length}"
    data_dir = os.path.join(output_root, mol_name)
    ensure_directory_exists(data_dir)

    # Stage 1: Hamiltonian
    hf_energy, n_elec, n_spin_orb = generate_hamiltonian(data_dir, geometry_label)

    # Stage 2: exact spectrum, overlaps of the reference state, and Hamiltonian moments
    calc_eigen(data_dir, mol_name, geometry_label, spin_label, hf_energy=hf_energy)
    eigen_path = os.path.join(data_dir, "eigen", f"{mol_name}_{geometry_label}_{n_elec}_{n_spin_orb}_{spin_label}_eigen.json")
    with open(eigen_path, "r") as eigen_file:
        eig_results = json.load(eigen_file)
    print(f"Lowest distinct energies with nonzero overlap: {eig_results['exact_energies_truncated'][:5]}")
    print(f"Their exact overlaps: {eig_results['overlaps_truncated'][:5]}")

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
