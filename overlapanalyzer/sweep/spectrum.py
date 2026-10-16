"""
Exact diagonalization (via eigen.calc_eigen) and automatic identification of the target eigenstate.

calc_eigen reports the full spectrum of the fixed-particle-number Hamiltonian, the overlap of the reference
state with every distinct energy level, and <Sz>, <N>, <S^2> of each eigenvector. Because the reference has a
definite spin, many low-lying levels can be invisible to it (zero overlap) or belong to another spin sector.
Choosing the physically relevant level was previously done by hand; select_target_level does it from the
configured strategy and reports diagnostics that make the choice auditable.

Everything in this module is exact (up to floating-point error): it comes from full diagonalization.
"""
from __future__ import annotations

import math
import os

import numpy as np

from overlapanalyzer.eigen import calc_eigen

# <S^2> of the reference states that calc_eigen builds: closed-shell singlet, open-shell singlet, Sz = +1 triplet
EXPECTED_REFERENCE_S2 = {"s0": 0.0, "s1": 0.0, "t1": 2.0}


class DimensionLimitExceeded(RuntimeError):
    """The dense diagonalization in calc_eigen would be too large; raise limits.max_dimension or use an active space."""


class TargetSelectionError(RuntimeError):
    """No eigenstate satisfies the configured target strategy."""


def eigen_json_path(system_dir: str, info, spin_label: str) -> str:
    # Mirrors calc_eigen's naming: '<mol>[CAS]_<geometry>_<n_elec>_<n_spin_orb>_<spin>_eigen.json'
    stem = info.mol_name + ("CAS" if info.uses_active_space else "")
    file_name = f"{stem}_{info.geometry_label}_{info.n_electrons}_{info.n_spin_orbitals}_{spin_label}_eigen.json"
    return os.path.join(system_dir, "eigen", file_name)


def run_exact_diagonalization(system, info, system_dir: str, max_dimension: int) -> str:
    dimension = info.hilbert_dimension
    if dimension > max_dimension:
        raise DimensionLimitExceeded(
            f"Fixed-N Hilbert space dimension {dimension} (C({info.n_spin_orbitals},{info.n_electrons})) exceeds "
            f"limits.max_dimension = {max_dimension}. calc_eigen diagonalizes a dense matrix "
            f"(~{16 * dimension ** 2 / 1e9:.1f} GB); use an active space or raise the limit.")
    calc_eigen(system_dir, info.mol_name, info.geometry_label, system.spin,
               hf_energy=info.hf_energy, threshold=system.overlap_threshold, create_CAS=info.uses_active_space)
    output_path = eigen_json_path(system_dir, info, system.spin)
    if not os.path.exists(output_path):
        raise RuntimeError(f"calc_eigen finished but {output_path} was not written.")
    return output_path


def build_level_table(eig_results: dict) -> list:
    """One entry per distinct energy level, with its overlap and spin character."""
    energies = eig_results["exact_energies_no_degen"]
    overlaps = eig_results["overlaps"]
    degeneracies = eig_results["degen"]
    exp_s2_per_vector = eig_results["symmetries"][2]

    levels = []
    vector_start = 0
    for level_index, (energy, overlap, degeneracy) in enumerate(zip(energies, overlaps, degeneracies)):
        level_s2_values = np.array(exp_s2_per_vector[vector_start:vector_start + degeneracy], dtype=float)
        mean_s2 = float(level_s2_values.mean())
        levels.append({
            "index": level_index,
            "energy": float(energy),
            "overlap": float(np.real(overlap)),
            "degeneracy": int(degeneracy),
            "s2": mean_s2,
            "s2_spread": float(level_s2_values.max() - level_s2_values.min()),
            "spin_multiplicity": math.sqrt(1 + 4 * max(mean_s2, 0.0)),  # 2S+1
        })
        vector_start += degeneracy
    return levels


def select_target_level(levels: list, target: dict, spin_label: str) -> dict:
    strategy = target["strategy"]
    min_overlap = target.get("min_overlap", 1e-6)

    if strategy == "lowest_with_overlap":
        candidates = [level for level in levels if level["overlap"] >= min_overlap]
    elif strategy == "max_overlap":
        candidates = [max(levels, key=lambda level: level["overlap"])]
    elif strategy == "index":
        if not 0 <= target["index"] < len(levels):
            raise TargetSelectionError(f"Target index {target['index']} is outside the {len(levels)} distinct levels.")
        candidates = [levels[target["index"]]]
    elif strategy == "lowest_in_sector":
        tolerance = target.get("tolerance", 1e-4)
        expected_s2 = EXPECTED_REFERENCE_S2[spin_label]
        candidates = [level for level in levels
                      if abs(level["s2"] - expected_s2) <= tolerance and level["s2_spread"] <= tolerance]
    else:
        raise TargetSelectionError(f"Unknown target strategy {strategy!r}")

    if not candidates:
        raise TargetSelectionError(f"No level satisfies target strategy {strategy!r} (min_overlap={min_overlap}).")
    return candidates[0]


def nearest_gaps(levels: list, target_level: dict, visibility_threshold: float) -> dict:
    """Energy gaps from the target to its neighbours: among all levels, and among levels the reference can see."""
    target_index = target_level["index"]
    target_energy = target_level["energy"]

    def gap_to(level_list):
        below = [level for level in level_list if level["index"] < target_index]
        above = [level for level in level_list if level["index"] > target_index]
        return (target_energy - below[-1]["energy"] if below else None,
                above[0]["energy"] - target_energy if above else None)

    visible_levels = [level for level in levels if level["overlap"] >= visibility_threshold]
    gap_below_any, gap_above_any = gap_to(levels)
    gap_below_visible, gap_above_visible = gap_to(visible_levels)
    return {
        "gap_below_any": gap_below_any,
        "gap_above_any": gap_above_any,
        "gap_below_visible": gap_below_visible,
        "gap_above_visible": gap_above_visible,
        "visibility_threshold": visibility_threshold,
        "n_visible_levels": len(visible_levels),
    }


def validate_spectrum(eig_results: dict, levels: list, spin_label: str) -> dict:
    """
    Exact identities every diagonalization must satisfy; violations indicate a bug or numerical trouble.

    - Moments: sum_i |<ref|i>|^2 x_i^k must equal the stored rescaled moment <ref|x^k|ref>, where x is the
      energy mapped by rescale_values exactly as calc_eigen maps H. This ties the spectrum to the moments
      that all bounds are computed from.
    - Completeness: overlaps over all levels sum to 1.
    - Reference spin: because H conserves S^2, the overlap-weighted <S^2> of the levels equals the reference's
      <S^2>, which must match the spin label (s0/s1 singlet, t1 triplet).
    """
    overlaps = np.array([level["overlap"] for level in levels])
    energies = np.array([level["energy"] for level in levels])
    rescale_lower, rescale_upper = eig_results["rescale_values"]
    rescaled_energies = (2 * energies - rescale_lower - rescale_upper) / (rescale_upper - rescale_lower)
    stored_moments = np.array(eig_results["moments_rescaled"], dtype=float)
    powers = np.arange(len(stored_moments))
    spectral_moments = (overlaps[:, None] * rescaled_energies[:, None] ** powers[None, :]).sum(axis=0)
    moment_error = float(np.max(np.abs(spectral_moments - stored_moments)))

    s2_from_overlaps = float(np.dot(overlaps, [level["s2"] for level in levels]))
    expected_s2 = EXPECTED_REFERENCE_S2[spin_label]
    return {
        "moment_consistency_max_error": moment_error,
        "moments_consistent": moment_error < 1e-8,
        "overlap_sum_error": float(abs(overlaps.sum() - 1.0)),
        "reference_s2_error": abs(s2_from_overlaps - expected_s2),
        "reference_spin_consistent": abs(s2_from_overlaps - expected_s2) < 1e-6,
    }


def summarize_spectrum(eig_results: dict, target: dict, spin_label: str) -> dict:
    """Exact spectral facts for one system and one target choice."""
    levels = build_level_table(eig_results)
    target_level = select_target_level(levels, target, spin_label)
    visibility_threshold = target.get("min_overlap", 1e-6)
    overlaps = np.array([level["overlap"] for level in levels])
    s2_values = np.array([level["s2"] for level in levels])
    hf_energy = eig_results.get("hf_energy")

    return {
        "target": {
            **target_level,
            "strategy": target["strategy"],
            "is_global_ground_state": target_level["index"] == 0,
            "energy_below_hf": (hf_energy - target_level["energy"]) if hf_energy is not None else None,
            **nearest_gaps(levels, target_level, visibility_threshold),
        },
        "global_ground_state": {key: levels[0][key] for key in ("energy", "overlap", "degeneracy", "s2")},
        "spectrum": {
            "n_levels": len(levels),
            "n_eigenvalues": int(sum(level["degeneracy"] for level in levels)),
            "energy_min": float(levels[0]["energy"]),
            "energy_max": float(levels[-1]["energy"]),
            "hf_energy": hf_energy,
        },
        "reference": {
            "spin_label": spin_label,
            "expected_s2": EXPECTED_REFERENCE_S2[spin_label],
            # Valid because H conserves S^2: sum_i |<ref|level_i>|^2 <S^2>_i = <ref|S^2|ref> for spin-pure levels
            "s2_from_overlaps": float(np.dot(overlaps, s2_values)),
            "overlap_sum": float(overlaps.sum()),
        },
        "validation": validate_spectrum(eig_results, levels, spin_label),
        "levels_preview": levels[:10],
    }


def truncated_index_of(eig_results: dict, energy: float):
    """Index of a level in calc_eigen's overlap-thresholded list, which is what the bound recipes index into."""
    truncated_energies = eig_results["exact_energies_truncated"]
    for index, candidate in enumerate(truncated_energies):
        if abs(candidate - energy) <= 1e-9 * max(1.0, abs(energy)):
            return index
    return None
