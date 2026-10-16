"""
End-to-end sweep tests on tiny molecules (H2, LiH in a small active space); about half a minute in total.
Energies are checked against PySCF's own FCI/CASCI, independently of the sweep's Hamiltonian and diagonalization.
"""
import csv
import json
import os

import pytest

pytest.importorskip("openfermionpyscf")

from overlapanalyzer.sweep.config import parse_sweep_config
from overlapanalyzer.sweep.runner import run_sweep, write_summaries
from overlapanalyzer.sweep.storage import SystemPaths

H2_GEOMETRIES = [0.74, 1.5]


def h2_config(extra_analyses=()):
    return parse_sweep_config({
        "sweep_name": "h2_test",
        "systems": [{"molecule": "h2", "geometries": H2_GEOMETRIES}],
        "analyses": [
            {"name": "ground", "modes": ["exact"], "poly_degrees": [2, 4]},
            {"name": "bad_index", "target": {"strategy": "index", "index": 99}, "modes": ["exact"], "poly_degrees": [2]},
            *extra_analyses,
        ],
    })


def load_result(output_dir, system, analysis):
    with open(SystemPaths(output_dir, system.system_id).result_json(analysis.analysis_id)) as handle:
        return json.load(handle)


def pyscf_reference_energy(atoms, basis, active_space=None):
    from pyscf import fci, gto, mcscf, scf
    molecule = gto.M(atom=[[symbol, position] for symbol, position in atoms], basis=basis, verbose=0)
    mean_field = scf.RHF(molecule).run()
    if active_space is None:
        return fci.FCI(mean_field).kernel()[0]
    casci = mcscf.CASCI(mean_field, active_space["active_spatial_orbitals"], active_space["active_electrons"])
    return casci.kernel()[0]


@pytest.fixture(scope="module")
def completed_h2_sweep(tmp_path_factory):
    output_dir = str(tmp_path_factory.mktemp("h2_sweep"))
    config = h2_config()
    outcomes = run_sweep(config, output_dir)
    return config, output_dir, outcomes


def test_successes_and_failures_are_isolated(completed_h2_sweep):
    config, output_dir, outcomes = completed_h2_sweep
    states = {(outcome.system_id, outcome.analysis_id): outcome.state for outcome in outcomes}
    ground, bad_index = config.analyses
    for system in config.systems:
        assert states[(system.system_id, ground.analysis_id)] == "complete"
        assert states[(system.system_id, bad_index.analysis_id)] == "failed"
        assert "outside" in load_result(output_dir, system, bad_index)["error"]


def test_exact_results_match_independent_fci(completed_h2_sweep):
    config, output_dir, _ = completed_h2_sweep
    ground = config.analyses[0]
    for system in config.systems:
        result = load_result(output_dir, system, ground)
        reference_energy = pyscf_reference_energy(result["system"]["atoms"], system.basis)
        assert result["exact"]["target"]["energy"] == pytest.approx(reference_energy, abs=1e-8)
        assert result["exact"]["reference"]["overlap_sum"] == pytest.approx(1.0, abs=1e-10)


def test_bounds_bracket_exact_overlap(completed_h2_sweep):
    config, output_dir, _ = completed_h2_sweep
    for system in config.systems:
        exact_mode = load_result(output_dir, system, config.analyses[0])["bounds"]["exact"]
        assert exact_mode["status"] == "complete"
        for row in exact_mode["degrees"]:
            assert row["brackets_exact"], row


def test_resume_skips_finished_work_and_retry_reruns_failures(completed_h2_sweep):
    config, output_dir, _ = completed_h2_sweep
    rerun = run_sweep(config, output_dir)
    assert {outcome.state for outcome in rerun} == {"cached-complete", "cached-failed"}

    retried = run_sweep(config, output_dir, retry_failed=True)
    assert sorted(outcome.state for outcome in retried) == ["cached-complete"] * 2 + ["failed"] * 2


def test_new_analysis_reuses_cached_system_stages(completed_h2_sweep):
    _, output_dir, _ = completed_h2_sweep
    extended = h2_config(extra_analyses=[{"name": "max_overlap", "target": {"strategy": "max_overlap"},
                                          "modes": ["exact"], "poly_degrees": [2]}])
    outcomes = run_sweep(extended, output_dir)
    new_analysis = extended.analyses[2]
    for system in extended.systems:
        result = load_result(output_dir, system, new_analysis)
        assert result["status"] == "complete"
        assert result["timing"]["hamiltonian"]["cached"] and result["timing"]["eigen"]["cached"]
    assert sum(outcome.state == "complete" for outcome in outcomes) == len(extended.systems)


def test_summaries_cover_every_experiment(completed_h2_sweep):
    config, output_dir, _ = completed_h2_sweep
    # Summaries describe the given config only, even though the directory also holds results of an extended config
    write_summaries(config, output_dir)
    with open(os.path.join(output_dir, "results.csv")) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == config.experiment_count
    assert {row["status"] for row in rows} == {"complete", "failed"}
    with open(os.path.join(output_dir, "bounds.csv")) as handle:
        assert len(list(csv.DictReader(handle))) == len(config.systems) * 2   # two degrees per complete experiment


def test_active_space_and_triplet_reference(tmp_path):
    active_space = {"active_electrons": 2, "active_spatial_orbitals": 3}
    config = parse_sweep_config({
        "systems": [{"molecule": "lih", "geometries": [1.6], "spin": ["s0", "t1"], "active_space": active_space}],
        "analyses": [{"name": "sector", "target": {"strategy": "lowest_in_sector"}, "modes": ["exact"], "poly_degrees": [2, 4]}],
    })
    run_sweep(config, str(tmp_path))
    singlet, triplet = config.systems
    singlet_result = load_result(str(tmp_path), singlet, config.analyses[0])
    triplet_result = load_result(str(tmp_path), triplet, config.analyses[0])

    # Singlet: frozen-core active-space ground state equals PySCF CASCI on the same RHF orbitals
    casci_energy = pyscf_reference_energy(singlet_result["system"]["atoms"], "sto-3g", active_space)
    assert singlet_result["exact"]["target"]["energy"] == pytest.approx(casci_energy, abs=1e-8)
    assert singlet_result["system"]["n_spin_orbitals"] == 6

    # Triplet: the selected level is a triplet, not the (invisible) singlet ground state
    target = triplet_result["exact"]["target"]
    assert target["s2"] == pytest.approx(2.0, abs=1e-6)
    assert not target["is_global_ground_state"]
    assert target["overlap"] > 0.5
