import pytest

from overlapanalyzer.sweep.spectrum import (TargetSelectionError, build_level_table, select_target_level,
                                            summarize_spectrum, truncated_index_of)


def synthetic_eig_results():
    # Levels: singlet ground state invisible to the reference, a visible triplet, then two visible singlets.
    # Eigenvectors are listed per degenerate copy, as calc_eigen stores them.
    return {
        "exact_energies_no_degen": [-2.0, -1.9, -1.5, -1.0],
        "overlaps": [0.0, 0.6, 0.3, 0.1],
        "degen": [1, 3, 1, 1],
        "symmetries": [[0.0, 1.0, 0.0, -1.0, 0.0, 0.0], [2, 2, 2, 2, 2, 2], [0.0, 2.0, 2.0, 2.0, 0.0, 0.0]],
        "exact_energies_truncated": [-1.9, -1.5, -1.0],
        "hf_energy": -1.7,
    }


def test_level_table_groups_degenerate_vectors():
    levels = build_level_table(synthetic_eig_results())
    assert [level["degeneracy"] for level in levels] == [1, 3, 1, 1]
    assert levels[1]["s2"] == pytest.approx(2.0)
    assert levels[1]["spin_multiplicity"] == pytest.approx(3.0)


@pytest.mark.parametrize("target, spin, expected_index", [
    ({"strategy": "lowest_with_overlap", "min_overlap": 1e-6}, "s0", 1),
    ({"strategy": "max_overlap"}, "s0", 1),
    ({"strategy": "index", "index": 3}, "s0", 3),
    ({"strategy": "lowest_in_sector", "tolerance": 1e-6}, "s0", 0),   # lowest singlet, even though invisible
    ({"strategy": "lowest_in_sector", "tolerance": 1e-6}, "t1", 1),
])
def test_target_strategies(target, spin, expected_index):
    levels = build_level_table(synthetic_eig_results())
    assert select_target_level(levels, target, spin)["index"] == expected_index


def test_unsatisfiable_target_raises():
    levels = build_level_table(synthetic_eig_results())
    with pytest.raises(TargetSelectionError):
        select_target_level(levels, {"strategy": "lowest_with_overlap", "min_overlap": 0.9}, "s0")
    with pytest.raises(TargetSelectionError):
        select_target_level(levels, {"strategy": "index", "index": 10}, "s0")


def test_summary_reports_gaps_and_reference_diagnostics():
    summary = summarize_spectrum(synthetic_eig_results(), {"strategy": "lowest_with_overlap", "min_overlap": 1e-6}, "s0")
    target = summary["target"]
    assert target["index"] == 1 and not target["is_global_ground_state"]
    assert target["gap_below_any"] == pytest.approx(0.1)
    assert target["gap_below_visible"] is None          # nothing visible below the target
    assert target["gap_above_visible"] == pytest.approx(0.4)
    assert target["energy_below_hf"] == pytest.approx(0.2)
    assert summary["reference"]["overlap_sum"] == pytest.approx(1.0)


def test_truncated_index_maps_energies():
    eig = synthetic_eig_results()
    assert truncated_index_of(eig, -1.5) == 1
    assert truncated_index_of(eig, -2.0) is None
