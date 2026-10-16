"""
Hamiltonian generation for sweep systems.

Two backends write the same file layout (ham_fer/<name>.data plus info.csv or CAS_info.csv), which is
the layout eigen.calc_eigen reads:

- "pyscf":   openfermionpyscf. Runs RHF (ROHF for triplet references) and builds the molecular Hamiltonian,
             optionally restricted to a frozen-core active space. Works with the public dependencies.
- "ham_gen": the original overlapanalyzer.ham_gen.gen_hamiltonian (CASCI integrals via PySCF + qcc_240202).
             Requires the non-public qcc_240202 package.

Both backends take the active space as the lowest (n_electrons - active_electrons) / 2 orbitals frozen and the
next active_spatial_orbitals orbitals active, which is PySCF's CASCI default used by ham_gen.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, asdict
from math import comb

from overlapanalyzer.read_ham import ensure_directory_exists
from overlapanalyzer.utils import get_molecular_data


@dataclass
class HamiltonianInfo:
    """What calc_eigen needs to find the Hamiltonian, plus facts worth recording."""
    mol_name: str           # calc_eigen's mol_name argument (calc_eigen appends 'CAS' itself for active spaces)
    geometry_label: str     # calc_eigen's geometry argument (middle part of the file name)
    n_electrons: int        # electrons in the (active) space that is diagonalized
    n_spin_orbitals: int    # spin orbitals = qubits of the (active) Hamiltonian
    hf_energy: float        # mean-field energy of the full molecule (RHF, or ROHF for t1)
    uses_active_space: bool
    hamiltonian_file: str
    atoms: list
    total_electrons: int
    total_spatial_orbitals: int

    @property
    def hilbert_dimension(self) -> int:
        """Dimension of the fixed-particle-number space that calc_eigen diagonalizes densely."""
        return comb(self.n_spin_orbitals, self.n_electrons)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["hilbert_dimension"] = self.hilbert_dimension
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "HamiltonianInfo":
        fields = {key: value for key, value in record.items() if key in cls.__dataclass_fields__}
        return cls(**fields)


def build_atoms(system) -> list:
    """Cartesian geometry in Angstrom as [(symbol, (x, y, z)), ...]."""
    if system.molecule == "custom":
        return [(symbol, tuple(position)) for symbol, position in system.atoms]
    if system.molecule == "h_chain":
        return [("H", (0.0, 0.0, i * system.geometry)) for i in range(system.n_atoms)]
    return [(symbol, tuple(float(coordinate) for coordinate in position))
            for symbol, position in get_molecular_data(system.molecule, system.geometry)]


def file_safe_molecule_name(system) -> str:
    """Molecule part of the Hamiltonian file name (matches ham_gen's naming for library molecules and chains)."""
    if system.molecule == "custom":
        return "".join(character if character.isalnum() else "-" for character in system.name)
    if system.molecule == "h_chain":
        return f"h{system.n_atoms}"
    return system.molecule


def remove_if_present(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def reference_multiplicity(spin_label: str) -> int:
    # 't1' references are built from ROHF orbitals of the triplet; 's0' and 's1' from RHF orbitals (as in ham_gen)
    return 3 if spin_label == "t1" else 1


def write_info_csv(system_dir: str, uses_active_space: bool, n_electrons: int, n_spin_orbitals: int) -> None:
    csv_name = "CAS_info.csv" if uses_active_space else "info.csv"
    with open(os.path.join(system_dir, csv_name), "w", newline="") as info_file:
        writer = csv.writer(info_file)
        writer.writerow(["N_electron", "N_spin_orb"])
        writer.writerow([n_electrons, n_spin_orbitals])


def resolve_active_space(system, total_electrons: int, total_spatial_orbitals: int):
    """Frozen-core and active orbital indices for an active space, validated against the actual molecule."""
    active_electrons = system.active_space["active_electrons"]
    active_orbitals = system.active_space["active_spatial_orbitals"]
    frozen_electrons = total_electrons - active_electrons
    if frozen_electrons < 0 or frozen_electrons % 2 != 0:
        raise ValueError(f"Active space with {active_electrons} electrons is incompatible with "
                         f"{total_electrons} total electrons (frozen electrons must be a non-negative even number).")
    n_core = frozen_electrons // 2
    if n_core + active_orbitals > total_spatial_orbitals:
        raise ValueError(f"Active space needs {n_core} core + {active_orbitals} active orbitals, "
                         f"but basis {system.basis} has only {total_spatial_orbitals}.")
    return list(range(n_core)), list(range(n_core, n_core + active_orbitals))


def generate_with_pyscf(system, system_dir: str) -> HamiltonianInfo:
    from openfermion import MolecularData, get_fermion_operator, save_operator
    from openfermionpyscf import run_pyscf

    atoms = build_atoms(system)
    multiplicity = reference_multiplicity(system.spin)
    molecule = MolecularData(atoms, system.basis, multiplicity, charge=system.charge,
                             filename=os.path.join(system_dir, "pyscf_molecule"))
    molecule = run_pyscf(molecule, run_scf=True)
    total_electrons = int(molecule.n_electrons)
    total_spatial_orbitals = int(molecule.n_orbitals)

    if system.active_space:
        core_indices, active_indices = resolve_active_space(system, total_electrons, total_spatial_orbitals)
        interaction_operator = molecule.get_molecular_hamiltonian(occupied_indices=core_indices, active_indices=active_indices)
        n_electrons = system.active_space["active_electrons"]
        n_spin_orbitals = 2 * len(active_indices)
    else:
        interaction_operator = molecule.get_molecular_hamiltonian()
        n_electrons = total_electrons
        n_spin_orbitals = 2 * total_spatial_orbitals

    uses_active_space = system.active_space is not None
    mol_name = file_safe_molecule_name(system)
    geometry_label = "fixed" if system.geometry is None else format(system.geometry, ".6g")
    file_stem = mol_name + ("CAS" if uses_active_space else "")
    hamiltonian_file = f"{file_stem}_{geometry_label}_{n_electrons}_{n_spin_orbitals}_{system.spin}.data"

    ham_dir = os.path.join(system_dir, "ham_fer")
    ensure_directory_exists(ham_dir)
    save_operator(get_fermion_operator(interaction_operator), file_name=hamiltonian_file,
                  data_directory=ham_dir, allow_overwrite=True, plain_text=True)
    write_info_csv(system_dir, uses_active_space, n_electrons, n_spin_orbitals)
    # run_pyscf caches all molecular integrals in an HDF5 file (O(N^4) in orbitals); nothing reads it again
    remove_if_present(molecule.filename + ".hdf5")

    return HamiltonianInfo(
        mol_name=mol_name, geometry_label=geometry_label, n_electrons=n_electrons,
        n_spin_orbitals=n_spin_orbitals, hf_energy=float(molecule.hf_energy), uses_active_space=uses_active_space,
        hamiltonian_file=os.path.join(ham_dir, hamiltonian_file), atoms=atoms,
        total_electrons=total_electrons, total_spatial_orbitals=total_spatial_orbitals,
    )


def generate_with_ham_gen(system, system_dir: str) -> HamiltonianInfo:
    """Delegate to the original ham_gen.gen_hamiltonian. Needs qcc_240202; not exercised without it."""
    try:
        from overlapanalyzer.ham_gen import gen_hamiltonian
    except ImportError as exc:
        raise RuntimeError("The ham_gen backend requires the non-public qcc_240202 package "
                           f"({exc}). Use backend \"pyscf\" instead.") from exc

    if system.molecule == "h_chain":
        mol_name, mode = f"h{system.n_atoms}", "h_chain"
    else:
        mol_name, mode = system.molecule, "from_MD"
    cas_kwargs = {}
    if system.active_space:
        cas_kwargs = {"create_CAS": True, "n_elec": system.active_space["active_electrons"],
                      "n_spin_orbs": 2 * system.active_space["active_spatial_orbitals"]}
    hf_energy = gen_hamiltonian(system_dir, mol_name, mode, small_atom_basis=system.basis,
                                R10=system.geometry * 10, spin=system.spin, **cas_kwargs)

    # ham_gen names the file with str(R10 / 10); read the name back rather than re-deriving the float formatting
    written_files = glob.glob(os.path.join(system_dir, "ham_fer", "*.data"))
    if len(written_files) != 1:
        raise RuntimeError(f"Expected one Hamiltonian file from ham_gen, found {len(written_files)}.")
    stem_parts = os.path.basename(written_files[0])[:-len(".data")].split("_")
    n_electrons, n_spin_orbitals = int(stem_parts[-3]), int(stem_parts[-2])
    geometry_label = stem_parts[-4]

    atoms = build_atoms(system)
    return HamiltonianInfo(
        mol_name=mol_name, geometry_label=geometry_label, n_electrons=n_electrons,
        n_spin_orbitals=n_spin_orbitals, hf_energy=float(hf_energy), uses_active_space=system.active_space is not None,
        hamiltonian_file=written_files[0], atoms=atoms,
        total_electrons=-1, total_spatial_orbitals=-1,  # Not reported by ham_gen
    )


def generate_hamiltonian(system, system_dir: str) -> HamiltonianInfo:
    ensure_directory_exists(system_dir)
    if system.backend == "pyscf":
        return generate_with_pyscf(system, system_dir)
    if system.backend == "ham_gen":
        return generate_with_ham_gen(system, system_dir)
    raise ValueError(f"Unknown backend {system.backend!r}")
