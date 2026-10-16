"""
Reproduce the README figures and result tables from the demonstration sweeps.

    .venv/bin/python scripts/make_figures.py

Runs (or resumes, if already cached) three example sweeps into sweeps/, copies their summary tables to
docs/results/<sweep>/, and draws each figure in a light and a dark variant into docs/images/.
All plotted numbers come straight from the sweep outputs.
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from overlapanalyzer.sweep.config import load_sweep_config
from overlapanalyzer.sweep.runner import run_sweep
from overlapanalyzer.sweep.spectrum import build_level_table, select_target_level
from overlapanalyzer.sweep.storage import StageStatus, SystemPaths

# overlapanalyzer.visualization sets global LaTeX/serif styles on import; reset to clean sans-serif defaults
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.usetex": False, "mathtext.default": "regular"})

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPOSITORY_ROOT, "docs", "images")
RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "docs", "results")
DEMO_SWEEPS = ["demo_bond_scan", "demo_bound_convergence", "lih_active_space", "validation_suite"]

# Validated categorical slots (blue, orange, aqua) and chart chrome, stepped separately for each theme
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"], "suffix": ""},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink_secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"], "suffix": "-dark"},
}
OVERLAP_FLOOR = 1e-10   # Overlaps below this are drawn on the floor line ("not visible to the reference")


# ---------------------------------------------------------------------------------------------
# Data

def run_demo_sweeps() -> dict:
    output_dirs = {}
    for sweep_name in DEMO_SWEEPS:
        config = load_sweep_config(os.path.join(REPOSITORY_ROOT, "examples", "sweeps", f"{sweep_name}.json"))
        output_dir = os.path.join(REPOSITORY_ROOT, "sweeps", sweep_name)
        run_sweep(config, output_dir, jobs=4)
        published_dir = os.path.join(RESULTS_DIR, sweep_name)
        os.makedirs(published_dir, exist_ok=True)
        for table in ("results.csv", "bounds.csv"):
            shutil.copy(os.path.join(output_dir, table), os.path.join(published_dir, table))
        output_dirs[sweep_name] = (config, output_dir)
    return output_dirs


def usable_bounds(bounds: pd.DataFrame) -> pd.DataFrame:
    """Keep bounds whose solver reported 'optimal' on both sides; inaccurate solves stay flagged in bounds.csv."""
    clean = bounds["solver_status"] == "lower=optimal;upper=optimal"
    return bounds[clean & bounds["lower"].notna() & bounds["upper"].notna()]


# ---------------------------------------------------------------------------------------------
# Styling

def styled_axes(axes, theme):
    axes.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1)
    axes.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    axes.set_axisbelow(True)
    axes.tick_params(colors=theme["muted"], labelcolor=theme["ink_secondary"], labelsize=9)
    axes.xaxis.label.set_color(theme["ink_secondary"])
    axes.yaxis.label.set_color(theme["ink_secondary"])


def new_figure(theme, columns, height=3.6):
    figure, axes_list = plt.subplots(1, columns, figsize=(4.4 * columns, height), dpi=200, squeeze=False)
    figure.patch.set_facecolor(theme["surface"])
    for axes in axes_list[0]:
        styled_axes(axes, theme)
    return figure, list(axes_list[0])


def add_legend(figure, handles, labels, theme, columns):
    legend = figure.legend(handles, labels, loc="lower center", ncol=columns, frameon=False, fontsize=9,
                           bbox_to_anchor=(0.5, -0.01))
    for text in legend.get_texts():
        text.set_color(theme["ink_secondary"])


def save(figure, name, theme):
    figure.savefig(os.path.join(IMAGE_DIR, f"{name}{theme['suffix']}.png"), facecolor=theme["surface"],
                   bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------
# Figures

def plot_bond_scan(results, bounds, theme):
    """Exact HF/ground-state overlap along a bond stretch, with exact-mode bounds at two polynomial degrees."""
    figure, axes_list = new_figure(theme, 2)
    blue, orange, aqua = theme["series"]
    bounds = usable_bounds(bounds[bounds["mode"] == "exact"])
    for axes, chain in zip(axes_list, ["h4", "h6"]):
        chain_results = results[results["system_id"].str.startswith(f"{chain}_chain")].sort_values("geometry")
        chain_bounds = bounds[bounds["system_id"].str.startswith(f"{chain}_chain")]
        for degree, color in ((4, orange), (8, aqua)):
            band = chain_bounds[chain_bounds["degree"] == degree].sort_values("geometry")
            axes.fill_between(band["geometry"], band["lower"], band["upper"], color=color, alpha=0.14, linewidth=0)
            axes.plot(band["geometry"], band["lower"], color=color, linewidth=1.6)
            axes.plot(band["geometry"], band["upper"], color=color, linewidth=1.6)
        axes.plot(chain_results["geometry"], chain_results["exact_target_overlap"], color=blue, linewidth=2,
                  solid_capstyle="round")
        axes.scatter(chain_results["geometry"], chain_results["exact_target_overlap"], s=22, color=blue,
                     edgecolor=theme["surface"], linewidth=1.2, zorder=3)
        axes.set_title(f"Linear H$_{chain[1]}$ chain, STO-3G", fontsize=11, loc="left", color=theme["ink"])
        axes.set_xlabel("H–H spacing (Å)")
        axes.set_ylim(-0.02, 1.02)
        last = chain_results.iloc[-1]
        axes.annotate(f"{last['exact_target_overlap']:.2f}", (last["geometry"], last["exact_target_overlap"]),
                      xytext=(4, 8), textcoords="offset points", fontsize=8.5, color=theme["ink_secondary"])
    axes_list[0].set_ylabel("Overlap |⟨HF|ground state⟩|²")
    handles = [plt.Line2D([], [], color=blue, linewidth=2, marker="o", markersize=5),
               plt.Rectangle((0, 0), 1, 1, facecolor=orange, alpha=0.35, edgecolor=orange),
               plt.Rectangle((0, 0), 1, 1, facecolor=aqua, alpha=0.35, edgecolor=aqua)]
    add_legend(figure, handles, ["Exact overlap (diagonalization)", "Bounds from moments, degree 4",
                                 "Bounds from moments, degree 8"], theme, 3)
    figure.subplots_adjust(bottom=0.24, wspace=0.18)
    save(figure, "bond_scan", theme)


def plot_bound_convergence(results, bounds, theme):
    """Lower/upper bounds versus polynomial degree for N2 near equilibrium and stretched."""
    figure, axes_list = new_figure(theme, 2)
    blue, orange, _ = theme["series"]
    clean = usable_bounds(bounds)
    for axes, geometry in zip(axes_list, [1.1, 1.6]):
        exact_value = results[results["geometry"] == geometry]["exact_target_overlap"].iloc[0]
        axes.axhline(exact_value, color=theme["ink_secondary"], linewidth=1)
        axes.annotate(f"exact overlap {exact_value:.3f}", (20.3, exact_value), xytext=(0, 6), textcoords="offset points",
                      fontsize=8.5, color=theme["ink_secondary"], ha="right")
        all_degrees = sorted(bounds["degree"].unique())
        for mode, color in (("exact", blue), ("approx", orange)):
            rows = clean[(clean["geometry"] == geometry) & (clean["mode"] == mode)].set_index("degree")
            rows = rows.reindex(all_degrees).reset_index()     # excluded degrees become gaps, not straight lines
            for side in ("lower", "upper"):
                axes.plot(rows["degree"], rows[side], color=color, linewidth=2, marker="o", markersize=4.5,
                          markeredgecolor=theme["surface"], markeredgewidth=1.2)
        axes.set_title(f"N$_2$ at {geometry} Å, CAS(6e, 6o)", fontsize=11, loc="left", color=theme["ink"])
        axes.set_xlabel("Polynomial degree (Hamiltonian moments used)")
        axes.set_xticks(range(2, 21, 2))
        axes.set_ylim(-0.02, 1.02)
    axes_list[0].set_ylabel("Bounds on |⟨HF|ground state⟩|²")
    handles = [plt.Line2D([], [], color=blue, linewidth=2, marker="o", markersize=4.5),
               plt.Line2D([], [], color=orange, linewidth=2, marker="o", markersize=4.5)]
    add_legend(figure, handles, ["exact mode (eigenvalues known)",
                                 "approx mode (eigenvalues known to 2% of neighbouring gaps)"], theme, 2)
    figure.subplots_adjust(bottom=0.24, wspace=0.18)
    save(figure, "bound_convergence", theme)


def load_levels(config, output_dir, geometry, spin):
    system = next(s for s in config.systems if s.geometry == geometry and s.spin == spin)
    eigen_file = StageStatus(SystemPaths(output_dir, system.system_id).status_json).get("eigen")["eigen_file"]
    with open(eigen_file) as handle:
        eig_results = json.load(handle)
    levels = build_level_table(eig_results)
    target = select_target_level(levels, {"strategy": "lowest_in_sector", "tolerance": 1e-4}, spin)
    return levels, target


def plot_spin_sector(config, output_dir, theme):
    """Which eigenstates a singlet vs triplet reference can see, and the level picked automatically."""
    figure, axes_list = new_figure(theme, 2, height=3.4)
    blue, orange, _ = theme["series"]
    for axes, spin, title in zip(axes_list, ["s0", "t1"], ["Singlet reference (closed-shell HF)", "Triplet reference (ROHF)"]):
        levels, target = load_levels(config, output_dir, 3.0, spin)
        for level in levels:
            color = blue if level["s2"] < 1 else orange
            overlap = max(level["overlap"], OVERLAP_FLOOR)
            axes.vlines(level["energy"], OVERLAP_FLOOR, overlap, color=color, linewidth=1.2, alpha=0.6)
            axes.scatter(level["energy"], overlap, s=26, color=color, edgecolor=theme["surface"], linewidth=1.2, zorder=3)
        axes.set_yscale("log")
        axes.set_ylim(OVERLAP_FLOOR / 3, 3)
        ground = levels[0]
        connector = {"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8}
        if target["index"] == ground["index"]:
            # Place the label in the widest empty energy gap so it does not sit on other levels' stems
            energies = sorted(level["energy"] for level in levels)
            gap_start = max(zip(energies, energies[1:]), key=lambda pair: pair[1] - pair[0])[0]
            axes.annotate(f"ground state =\nauto-selected target\noverlap {target['overlap']:.2f}",
                          (target["energy"], target["overlap"]), xytext=(gap_start + 0.02, 1e-5), textcoords="data",
                          fontsize=8.5, color=theme["ink"], arrowprops=connector)
        else:
            axes.annotate("true ground state\n(invisible to this reference)", (ground["energy"], OVERLAP_FLOOR),
                          xytext=(16, 44), textcoords="offset points", fontsize=8.5, color=theme["ink_secondary"],
                          arrowprops=connector)
            axes.annotate(f"auto-selected target: lowest triplet\noverlap {target['overlap']:.2f}",
                          (target["energy"], target["overlap"]), xytext=(16, -40), textcoords="offset points",
                          fontsize=8.5, color=theme["ink"], arrowprops=connector)
        axes.set_title(title, fontsize=11, loc="left", color=theme["ink"], pad=10)
        axes.set_xlabel("Energy (Hartree)")
    axes_list[0].set_ylabel("Overlap with reference (log)")
    axes_list[0].text(0.0, -0.36, f"Levels with overlap below {OVERLAP_FLOOR:g} are drawn on the bottom line "
                      "(invisible to the reference).", transform=axes_list[0].transAxes, fontsize=8,
                      color=theme["muted"])
    handles = [plt.Line2D([], [], color=blue, linestyle="", marker="o", markersize=6),
               plt.Line2D([], [], color=orange, linestyle="", marker="o", markersize=6)]
    add_legend(figure, handles, ["singlet level (S² = 0)", "triplet level (S² = 2)"], theme, 2)
    figure.subplots_adjust(bottom=0.3, wspace=0.18)
    save(figure, "spin_sector", theme)


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    sweeps = run_demo_sweeps()
    bond_scan = pd.read_csv(os.path.join(RESULTS_DIR, "demo_bond_scan", "results.csv")), \
        pd.read_csv(os.path.join(RESULTS_DIR, "demo_bond_scan", "bounds.csv"))
    convergence = pd.read_csv(os.path.join(RESULTS_DIR, "demo_bound_convergence", "results.csv")), \
        pd.read_csv(os.path.join(RESULTS_DIR, "demo_bound_convergence", "bounds.csv"))
    for theme in THEMES.values():
        plot_bond_scan(*bond_scan, theme)
        plot_bound_convergence(*convergence, theme)
        plot_spin_sector(*sweeps["lih_active_space"], theme)
    print(f"Figures written to {IMAGE_DIR}; tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
