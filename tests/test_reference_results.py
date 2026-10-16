"""
Scientific validation against established results computed outside this pipeline:

- Psi4 HF/FCI energies shipped with OpenFermion (openfermion/testing/data), independent of PySCF;
- reference-state overlaps |<HF|FCI>|^2 from PySCF's own FCI/CASCI wavefunctions, independent of
  openfermion operators and eigen.calc_eigen.
"""
import os

import numpy as np
import pytest

pytest.importorskip("openfermionpyscf")

from conftest import load_result

PSI4_TOLERANCE = 1e-6       # Psi4 vs PySCF agree to ~1e-9 Ha; allow for SCF convergence thresholds


def psi4_reference(file_stem):
    import openfermion
    from openfermion import MolecularData
    data_dir = os.path.join(os.path.dirname(openfermion.__file__), "testing", "data")
    if not os.path.exists(os.path.join(data_dir, file_stem + ".hdf5")):
        pytest.skip(f"OpenFermion reference data {file_stem} not available")
    molecule = MolecularData(filename=os.path.join(data_dir, file_stem))
    molecule.load()
    return molecule.hf_energy, molecule.fci_energy


def pyscf_reference(atoms, basis, charge=0, active_space=None):
    """Ground-state energy and |<HF|ground>|^2 from PySCF FCI (or CASCI) on RHF orbitals."""
    from pyscf import fci, gto, mcscf, scf
    molecule = gto.M(atom=[[symbol, position] for symbol, position in atoms], basis=basis, charge=charge, verbose=0)
    mean_field = scf.RHF(molecule).run()
    if active_space is None:
        energy, ci_vector = fci.FCI(mean_field).kernel()
    else:
        casci = mcscf.CASCI(mean_field, active_space["active_spatial_orbitals"], active_space["active_electrons"])
        casci.kernel()
        energy, ci_vector = casci.e_tot, casci.ci
    # The (0, 0) element is the coefficient of the HF determinant in the MO-basis CI vector
    return energy, float(np.asarray(ci_vector)[0, 0] ** 2)


GROUND = {"name": "ground", "modes": ["exact"], "poly_degrees": [2, 4]}


@pytest.mark.parametrize("bond_length", [0.5, 0.7414, 1.0, 1.5, 2.5])
def test_h2_sto3g_matches_psi4(run_config, bond_length):
    config, output_dir, _ = run_config({"systems": [{"molecule": "h2", "geometries": [bond_length]}], "analyses": [GROUND]})
    result = load_result(output_dir, config.systems[0], config.analyses[0])
    psi4_hf, psi4_fci = psi4_reference(f"H2_sto-3g_singlet_{bond_length}")
    assert result["system"]["hf_energy"] == pytest.approx(psi4_hf, abs=PSI4_TOLERANCE)
    assert result["exact"]["target"]["energy"] == pytest.approx(psi4_fci, abs=PSI4_TOLERANCE)


def test_h2_631g_and_lih_full_space_match_psi4(run_config):
    config, output_dir, _ = run_config({
        "systems": [{"molecule": "h2", "geometries": [0.75], "basis": "6-31g"},
                    {"molecule": "lih", "geometries": [1.45]}],
        "analyses": [GROUND],
    })
    for system, stem in zip(config.systems, ["H2_6-31g_singlet_0.75", "H1-Li1_sto-3g_singlet_1.45"]):
        result = load_result(output_dir, system, config.analyses[0])
        psi4_hf, psi4_fci = psi4_reference(stem)
        assert result["system"]["hf_energy"] == pytest.approx(psi4_hf, abs=PSI4_TOLERANCE)
        assert result["exact"]["target"]["energy"] == pytest.approx(psi4_fci, abs=PSI4_TOLERANCE)


CASES = [
    # (system entry, active space or None)
    ({"molecule": "h2", "geometries": [3.0]}, None),                         # dissociation: near singlet/triplet degeneracy
    ({"molecule": "h_chain", "n_atoms": 4, "geometries": [1.8]}, None),       # strongly correlated chain
    ({"molecule": "h4", "geometries": [90.0]}, None),                         # square H4: degenerate frontier orbitals
    ({"name": "heh_cation", "atoms": [["He", [0, 0, 0]], ["H", [0, 0, 0.772]]], "charge": 1}, None),
    ({"molecule": "beh2", "geometries": [1.33]}, {"active_electrons": 4, "active_spatial_orbitals": 4}),
    ({"molecule": "h2o", "geometries": [0.96]}, {"active_electrons": 4, "active_spatial_orbitals": 4}),
    ({"molecule": "n2", "geometries": [1.1]}, {"active_electrons": 6, "active_spatial_orbitals": 6}),
]


@pytest.mark.parametrize("system_entry, active_space", CASES, ids=["h2-3.0", "h4-chain", "h4-square", "heh+", "beh2-cas", "h2o-cas", "n2-cas"])
def test_energy_and_overlap_match_pyscf_wavefunction(run_config, system_entry, active_space):
    entry = {**system_entry, **({"active_space": active_space} if active_space else {})}
    config, output_dir, _ = run_config({"systems": [entry], "analyses": [GROUND]})
    result = load_result(output_dir, config.systems[0], config.analyses[0])
    assert result["status"] == "complete", result.get("error")

    reference_energy, reference_overlap = pyscf_reference(result["system"]["atoms"], "sto-3g",
                                                          charge=config.systems[0].charge, active_space=active_space)
    target = result["exact"]["target"]
    assert target["energy"] == pytest.approx(reference_energy, abs=1e-8)
    assert target["overlap"] == pytest.approx(reference_overlap, abs=1e-8)
    assert target["s2"] == pytest.approx(0.0, abs=1e-6)

    # Built-in self-checks and bound validity hold on every case
    assert result["exact"]["validation"]["moments_consistent"]
    assert result["exact"]["validation"]["reference_spin_consistent"]
    assert all(row["brackets_exact"] for row in result["bounds"]["exact"]["degrees"])


def test_open_shell_singlet_reference_is_a_singlet(run_config):
    """
    Regression test for calc_eigen's 's1' reference, which used to be the Sz=0 triplet.
    The HOMO->LUMO open-shell singlet has a different spatial symmetry from the closed-shell ground state, so it
    sees no overlap there; its lowest visible level must be an excited singlet.
    """
    config, output_dir, _ = run_config({
        "systems": [{"molecule": "h2", "geometries": [0.7414], "spin": "s1"},
                    {"molecule": "h_chain", "n_atoms": 4, "geometries": [1.2], "spin": "s1"}],
        "analyses": [{"name": "visible", "modes": ["exact"], "poly_degrees": [2, 4]}],
    })
    for system in config.systems:
        result = load_result(output_dir, system, config.analyses[0])
        assert result["exact"]["reference"]["s2_from_overlaps"] == pytest.approx(0.0, abs=1e-8)
        assert result["exact"]["validation"]["reference_spin_consistent"]
        target = result["exact"]["target"]
        assert target["s2"] == pytest.approx(0.0, abs=1e-6)
        assert not target["is_global_ground_state"] and target["overlap"] > 0.1
