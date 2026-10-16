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


def finite_or_none(value):
    """Solvers report a missing bound as None (cvxpy) or NaN (linprog); normalise both to None."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def solver_statuses(raw_entry: dict | None) -> dict:
    """Solver status of the lower- and upper-bound problems: cvxpy status strings, or linprog success/message."""
    statuses = {}
    if not raw_entry:
        return statuses
    for side, problem_key, result_key in (("lower", "LB_problem", "LB_result"), ("upper", "UB_problem", "UB_result")):
        problem = raw_entry.get(problem_key)
        result = raw_entry.get(result_key)
        if problem is not None and getattr(problem, "status", None) is not None:
            statuses[side] = str(problem.status)                       # cvxpy: optimal, optimal_inaccurate, infeasible, ...
        elif result is not None and hasattr(result, "success"):
            statuses[side] = "optimal" if result.success else f"failed: {getattr(result, 'message', '')}".strip()
    return statuses


def summarize_degree_results(mode: str, payload: dict, exact_value) -> tuple[list, str]:
    """
    One row per polynomial degree with bounds, margins against the exact value, and solver status.
    Returns the rows and the mode status: 'complete' (every degree bounded on both sides), 'partial', or 'no_solution'.
    """
    tolerance = BRACKET_TOLERANCE[mode]
    raw_by_degree = {entry.get("degree"): entry for entry in payload.get("raw_solver_output") or [] if isinstance(entry, dict)}
    rows = []
    for entry in payload["degree_results"]:
        degree = int(entry["degree"])
        lower, upper = finite_or_none(entry.get("lower")), finite_or_none(entry.get("upper"))
        row = {"degree": degree, "lower": lower, "upper": upper,
               "width": (upper - lower) if lower is not None and upper is not None else None,
               "solver_status": solver_statuses(raw_by_degree.get(degree))}
        if exact_value is not None and lower is not None and upper is not None:
            row["margin_lower"] = exact_value - lower    # >= 0 when the lower bound is valid
            row["margin_upper"] = upper - exact_value    # >= 0 when the upper bound is valid
            row["brackets_exact"] = bool(row["margin_lower"] >= -tolerance and row["margin_upper"] >= -tolerance)
        rows.append(row)

    bounded_rows = [row for row in rows if row["width"] is not None]
    if rows and len(bounded_rows) == len(rows):
        mode_status = "complete"
    elif bounded_rows:
        mode_status = "partial"
    else:
        mode_status = "no_solution"
    return rows, mode_status


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
    degree_rows, mode_status = summarize_degree_results(mode, payload, exact_value)
    mode_record = {
        "status": mode_status,
        "description": MODE_DESCRIPTIONS[mode],
        "solver": "scipy-highs" if mode == "exact" else analysis.solver,
        "exact_reference_value": exact_value,
        "bracket_tolerance": BRACKET_TOLERANCE[mode],
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
