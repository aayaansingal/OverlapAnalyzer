"""
Polynomial overlap bounds for a selected target level, computed with the upstream OverlapEstimator.

Bounds are estimates built only from Hamiltonian moments <ref|H^k|ref> plus mode-specific spectral knowledge.
The exact overlap is recorded next to every bound as a *reference value*, together with the margins, so the
quality of a bound (and any solver inaccuracy) can be judged. The exact value is never fed into the bounds.

Each mode runs in its own compute_bounds call, so a solver failure in one mode does not lose the others.
"""
from __future__ import annotations

import contextlib
import copy
import os

import numpy as np

from overlapanalyzer.polynomial_majorisation_LB import OverlapEstimator

MODE_DESCRIPTIONS = {
    "exact": "Overlap with the target level; constraints use the exact eigenvalues (LP via scipy linprog).",
    "approx": "Overlap with the target level; eigenvalues known only within +/- bound_extension (SDP via cvxpy).",
    "below": "Total overlap with all levels below energy_threshold; only the threshold and spectral range used (SDP via cvxpy).",
}

# Tolerance for the 'brackets_exact' diagnostic: LP solutions are tight to ~1e-9, SCS solutions to ~1e-4
BRACKET_TOLERANCE = {"exact": 1e-7, "approx": 1e-4, "below": 1e-4}


def build_recipe(analysis, mode: str, eig_results: dict, target_truncated_index: int) -> dict:
    recipe = {
        **analysis.recipe_overrides,
        "mode": mode,
        "eig_results": eig_results,
        "poly_degrees": list(analysis.poly_degrees),
        "target_indices": [target_truncated_index],
        "b_values": [1],
        "solver": analysis.solver,
    }
    if mode == "approx":
        recipe["bound_extension"] = analysis.bound_extension
    if mode == "below":
        recipe["energy_threshold"] = analysis.energy_threshold
    return recipe


def reference_value(mode: str, payload: dict, target_overlap: float):
    """The exact value of the quantity a mode bounds."""
    if mode == "below":
        return payload.get("exact_overlap")
    return target_overlap


def compute_mode_bounds(analysis, mode: str, eig_results: dict, target_truncated_index: int,
                        target_overlap: float, analysis_dir: str) -> dict:
    estimator = OverlapEstimator()
    recipe = build_recipe(analysis, mode, copy.deepcopy(eig_results), target_truncated_index)
    plot_kwargs = {}
    if analysis.save_plots:
        plot_kwargs = {"visualize": True, "save_plots": True, "save_dir": analysis_dir}
    payload = estimator.compute_bounds(recipe, **plot_kwargs)
    estimator.save_results(payload, os.path.join(analysis_dir, f"bounds_{mode}.json"), fmt="json", distilled=True)

    exact_value = reference_value(mode, payload, target_overlap)
    tolerance = BRACKET_TOLERANCE[mode]
    degree_rows = []
    for entry in payload["degree_results"]:
        lower, upper = float(entry["lower"]), float(entry["upper"])
        row = {"degree": int(entry["degree"]), "lower": lower, "upper": upper, "width": upper - lower}
        if exact_value is not None and np.isfinite(lower) and np.isfinite(upper):
            row["margin_lower"] = exact_value - lower    # >= 0 when the lower bound is valid
            row["margin_upper"] = upper - exact_value    # >= 0 when the upper bound is valid
            row["brackets_exact"] = bool(row["margin_lower"] >= -tolerance and row["margin_upper"] >= -tolerance)
        degree_rows.append(row)

    mode_record = {
        "status": "complete",
        "description": MODE_DESCRIPTIONS[mode],
        "exact_reference_value": exact_value,
        "bracket_tolerance": tolerance,
        "degrees": degree_rows,
    }
    if mode == "approx":
        mode_record["bound_extension"] = analysis.bound_extension
    if mode == "below":
        mode_record["energy_threshold"] = analysis.energy_threshold
    return mode_record


@contextlib.contextmanager
def propagate_solver_interrupts():
    """
    SCS traps Ctrl-C during a solve and returns status SIGINT, which cvxpy reports as an ordinary SolverError.
    Inside this context such a result is re-raised as KeyboardInterrupt, so an interrupted sweep stops and the
    experiment stays pending, instead of being recorded (and cached) as a solver failure.
    """
    try:
        import scs
    except ImportError:
        scs = None
    if scs is None:
        yield
        return

    original_solve = scs.solve

    def solve_with_interrupts(*args, **kwargs):
        results = original_solve(*args, **kwargs)
        if results.get("info", {}).get("status_val") == scs.SIGINT:
            raise KeyboardInterrupt("SCS solve interrupted")
        return results

    scs.solve = solve_with_interrupts   # cvxpy looks up scs.solve at call time
    try:
        yield
    finally:
        scs.solve = original_solve


def compute_all_bounds(analysis, eig_results: dict, target_truncated_index: int, target_overlap: float,
                       analysis_dir: str) -> dict:
    bounds_by_mode = {}
    with propagate_solver_interrupts():
        for mode in analysis.modes:
            try:
                bounds_by_mode[mode] = compute_mode_bounds(analysis, mode, eig_results, target_truncated_index,
                                                           target_overlap, analysis_dir)
            except Exception as exc:   # One failing solver must not discard the other modes
                bounds_by_mode[mode] = {"status": "failed", "description": MODE_DESCRIPTIONS[mode],
                                        "error": f"{type(exc).__name__}: {exc}"}
    return bounds_by_mode
