"""
Mathematical properties of the upstream bound algorithms (OverlapEstimator), checked on random synthetic spectra
and on a real molecule, plus the sweep's handling of missing/NaN solver results.

Properties:
- validity: lower <= exact overlap <= upper (exact eigenvalues in 'exact' mode; eigenvalues known within
  +/- bound_extension in 'approx' mode; cumulative overlap below a threshold in 'below' mode);
- monotonicity in 'exact' mode: degree-d polynomials are a subset of degree-(d+1) polynomials, so the optimal
  lower bound cannot decrease and the optimal upper bound cannot increase with the degree.
"""
import numpy as np
import pytest

from overlapanalyzer.polynomial_majorisation_LB import OverlapEstimator
from overlapanalyzer.sweep.bounds import solver_statuses, summarize_degree_results

from conftest import load_result

MAX_POWER = 12


def synthetic_eig_results(seed, n_levels=8):
    """A random spectrum and overlap distribution in the exact dictionary format calc_eigen writes."""
    rng = np.random.default_rng(seed)
    energies = np.sort(rng.uniform(-2.0, 0.0, n_levels))
    while np.min(np.diff(energies)) < 0.05:              # keep levels resolvable
        energies = np.sort(rng.uniform(-2.0, 0.0, n_levels))
    overlaps = rng.dirichlet(np.full(n_levels, 0.7))
    rescale_lower, rescale_upper = energies[-1], energies[0]  # calc_eigen's default (E_max, E_0) ordering
    rescaled = (2 * energies - rescale_lower - rescale_upper) / (rescale_upper - rescale_lower)
    return {
        "exact_energies_no_degen": energies.tolist(),
        "exact_energies_truncated": energies.tolist(),
        "overlaps": overlaps.tolist(),
        "overlaps_truncated": overlaps.tolist(),
        "rescale_values": [rescale_lower, rescale_upper],
        "moments": [float(np.dot(overlaps, energies ** k)) for k in range(MAX_POWER + 1)],
        "moments_rescaled": [float(np.dot(overlaps, rescaled ** k)) for k in range(MAX_POWER + 1)],
    }


def bounds_for(recipe):
    payload = OverlapEstimator().compute_bounds(recipe)
    return [(entry["degree"], entry["lower"], entry["upper"]) for entry in payload["degree_results"]], payload


@pytest.mark.parametrize("seed", range(12))
def test_exact_mode_bounds_are_valid_and_monotone(seed):
    eig = synthetic_eig_results(seed)
    target_index = int(np.argmax(eig["overlaps"]))
    exact_overlap = eig["overlaps"][target_index]
    rows, _ = bounds_for({"mode": "exact", "eig_results": eig, "target_indices": [target_index], "b_values": [1],
                          "poly_degrees": [2, 4, 6, 8, 10]})
    for _, lower, upper in rows:
        assert lower - 1e-7 <= exact_overlap <= upper + 1e-7
    lowers = [lower for _, lower, _ in rows]
    uppers = [upper for _, _, upper in rows]
    assert all(later >= earlier - 1e-8 for earlier, later in zip(lowers, lowers[1:]))
    assert all(later <= earlier + 1e-8 for earlier, later in zip(uppers, uppers[1:]))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("target_indices", [[0], [2], [3], [5], [1, 4]])
def test_approx_mode_bounds_are_valid_for_every_target_position(seed, target_indices):
    """Regression: targets after merged non-target regions used to be mapped to the wrong region (or crash)."""
    eig = synthetic_eig_results(seed, n_levels=6)
    exact_overlap = sum(eig["overlaps"][index] for index in target_indices)
    rows, payload = bounds_for({"mode": "approx", "eig_results": eig, "target_indices": target_indices,
                          "b_values": [1] * len(target_indices), "poly_degrees": [4, 8], "bound_extension": 0.02,
                          "solver": "SCS"})
    statuses = {entry["degree"]: solver_statuses(entry) for entry in payload["raw_solver_output"]}
    for degree, lower, upper in rows:
        if statuses[degree] != {"lower": "optimal", "upper": "optimal"}:
            continue    # SCS did not converge; the sweep flags these results instead of trusting them
        assert lower - 1e-4 <= exact_overlap <= upper + 1e-4, (degree, lower, upper, exact_overlap, statuses[degree])


@pytest.mark.parametrize("target_index", [1, 3, 7])
def test_exact_mode_bounds_are_valid_for_non_ground_targets(target_index):
    eig = synthetic_eig_results(seed=5)
    rows, _ = bounds_for({"mode": "exact", "eig_results": eig, "target_indices": [target_index], "b_values": [1],
                          "poly_degrees": [4, 8]})
    for _, lower, upper in rows:
        assert lower - 1e-7 <= eig["overlaps"][target_index] <= upper + 1e-7


@pytest.mark.parametrize("seed", range(3))
def test_below_mode_bounds_cumulative_overlap(seed):
    eig = synthetic_eig_results(seed, n_levels=6)
    energies = eig["exact_energies_no_degen"]
    threshold = 0.5 * (energies[2] + energies[3])      # three levels lie below
    cumulative_overlap = sum(eig["overlaps"][:3])
    rows, payload = bounds_for({"mode": "below", "eig_results": eig, "energy_threshold": threshold,
                                "poly_degrees": [4, 8], "solver": "SCS"})
    assert payload["exact_overlap"] == pytest.approx(cumulative_overlap, abs=1e-12)
    for _, lower, upper in rows:
        assert lower - 1e-4 <= cumulative_overlap <= upper + 1e-4


def test_exact_mode_monotone_on_a_molecule(run_config):
    config, output_dir, _ = run_config({
        "systems": [{"molecule": "h_chain", "n_atoms": 4, "geometries": [1.6]}],
        "analyses": [{"name": "deep", "modes": ["exact"], "poly_degrees": [2, 4, 6, 8, 10, 12, 16, 20]}],
    })
    rows = load_result(output_dir, config.systems[0], config.analyses[0])["bounds"]["exact"]["degrees"]
    assert all(row["brackets_exact"] and row["solver_status"] == {"lower": "optimal", "upper": "optimal"} for row in rows)
    assert all(b["lower"] >= a["lower"] - 1e-8 and b["upper"] <= a["upper"] + 1e-8 for a, b in zip(rows, rows[1:]))
    assert rows[-1]["width"] < 1e-3      # high degree pins the overlap down tightly


def test_missing_and_nan_solver_results_are_kept_per_degree():
    payload = {"degree_results": [
        {"degree": 2, "lower": 0.1, "upper": 0.9},
        {"degree": 4, "lower": None, "upper": 0.8},           # cvxpy: no solution for the lower-bound problem
        {"degree": 6, "lower": float("nan"), "upper": float("nan")},  # linprog failure
    ]}
    rows, status = summarize_degree_results("approx", payload, exact_value=0.5)
    assert status == "partial"
    assert rows[0]["brackets_exact"] and rows[0]["width"] == pytest.approx(0.8)
    assert rows[1]["lower"] is None and rows[1]["width"] is None and "brackets_exact" not in rows[1]
    assert rows[2]["upper"] is None

    _, status = summarize_degree_results("approx", {"degree_results": [{"degree": 2, "lower": None, "upper": None}]}, 0.5)
    assert status == "no_solution"
