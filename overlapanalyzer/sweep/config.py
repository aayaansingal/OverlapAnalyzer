"""
Sweep configuration: loading, validation and expansion into concrete system and analysis specs.

Validation happens entirely up front, so a typo in a config fails before any computation starts.
Each spec has a stable content hash, which is what makes resuming and result caching reliable:
a spec whose parameters change gets a new identity and is recomputed, while unchanged specs are reused.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field, asdict

# Molecules with a one-parameter geometry in overlapanalyzer.utils.get_molecular_data.
# The meaning of the geometry parameter is molecule-specific (bond length in Angstrom for most;
# the H-origin-H angle in degrees for 'h4'), exactly as defined there.
LIBRARY_MOLECULES = {"h2", "lih", "beh2", "h2o", "n2", "h4", "h4_linear", "nh3", "h2x2"}
GENERATED_MOLECULES = {"h_chain"}   # Linear chain of n_atoms hydrogens; geometry = spacing in Angstrom
SPIN_LABELS = {"s0", "t1", "s1"}    # Reference states understood by eigen.calc_eigen
BACKENDS = {"pyscf", "ham_gen"}
BOUND_MODES = {"exact", "approx", "below"}
TARGET_STRATEGIES = {"lowest_with_overlap", "max_overlap", "index", "lowest_in_sector"}

# eigen.calc_eigen stores Hamiltonian moments up to this power, which caps the polynomial degree
MAX_MOMENT_DEGREE = 40

SYSTEM_KEYS = {"molecule", "name", "geometries", "geometry", "basis", "spin", "active_space",
               "backend", "n_atoms", "atoms", "charge", "overlap_threshold"}
ANALYSIS_KEYS = {"name", "target", "modes", "poly_degrees", "bound_extension", "energy_threshold",
                 "solver", "save_plots", "recipe_overrides"}
TOP_LEVEL_KEYS = {"sweep_name", "description", "output_dir", "defaults", "systems", "analyses", "limits"}
LIMIT_KEYS = {"max_dimension"}


class ConfigError(ValueError):
    """Raised for any invalid sweep configuration; the message names the offending entry."""


def stable_hash(payload: dict, length: int = 10) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def format_geometry(value) -> str:
    return format(float(value), ".6g")


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


@dataclass(frozen=True)
class SystemSpec:
    """One molecular system: everything that determines the Hamiltonian, the reference state and the spectrum."""
    molecule: str
    geometry: float | None
    basis: str
    spin: str
    backend: str = "pyscf"
    active_space: dict | None = None    # {"active_electrons": int, "active_spatial_orbitals": int}
    n_atoms: int | None = None          # h_chain only
    atoms: tuple | None = None          # custom fixed geometry: ((symbol, (x, y, z)), ...)
    charge: int = 0
    overlap_threshold: float = 0.0      # passed to calc_eigen; levels with smaller overlap are dropped from bound constraints
    name: str | None = None             # display name; required for custom atoms

    def identity(self) -> dict:
        identity = asdict(self)
        identity.pop("name")    # Renaming a system does not change its physics
        return identity

    @property
    def label(self) -> str:
        base = self.name or (f"h{self.n_atoms}_chain" if self.molecule == "h_chain" else self.molecule)
        parts = [base]
        if self.geometry is not None:
            parts.append(f"g{format_geometry(self.geometry)}")
        parts += [self.basis, self.spin]
        if self.active_space:
            parts.append(f"cas{self.active_space['active_electrons']}e{self.active_space['active_spatial_orbitals']}o")
        if self.charge:
            parts.append(f"q{self.charge}")
        return slugify("_".join(parts))

    @property
    def system_id(self) -> str:
        return f"{self.label}-{stable_hash(self.identity())}"


@dataclass(frozen=True)
class AnalysisSpec:
    """How to pick the target eigenstate and which overlap bounds to compute for it."""
    name: str
    target: dict
    modes: tuple
    poly_degrees: tuple
    bound_extension: float | None = None
    energy_threshold: float | str | None = None
    solver: str = "SCS"
    save_plots: bool = False
    recipe_overrides: dict = field(default_factory=dict)

    def identity(self) -> dict:
        identity = asdict(self)
        identity.pop("save_plots")  # Plotting does not change the numbers
        return identity

    @property
    def analysis_id(self) -> str:
        return f"{slugify(self.name)}-{stable_hash(self.identity())}"


@dataclass(frozen=True)
class SweepConfig:
    sweep_name: str
    systems: tuple
    analyses: tuple
    max_dimension: int
    output_dir: str | None
    raw: dict

    @property
    def experiment_count(self) -> int:
        return len(self.systems) * len(self.analyses)


# ---------------------------------------------------------------------------------------------
# Loading

def load_sweep_config(path: str) -> SweepConfig:
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return parse_sweep_config(raw_config)


def parse_sweep_config(raw_config: dict) -> SweepConfig:
    if not isinstance(raw_config, dict):
        raise ConfigError("The sweep configuration must be a JSON object.")
    reject_unknown_keys(raw_config, TOP_LEVEL_KEYS, "top level")

    defaults = raw_config.get("defaults", {})
    reject_unknown_keys(defaults, SYSTEM_KEYS, "defaults")

    raw_systems = raw_config.get("systems")
    if not raw_systems:
        raise ConfigError("'systems' must be a non-empty list.")
    systems = []
    for position, raw_system in enumerate(raw_systems):
        context = f"systems[{position}]"
        reject_unknown_keys(raw_system, SYSTEM_KEYS, context)
        systems.extend(expand_system_entry({**defaults, **raw_system}, context))
    systems = deduplicate(systems, key=lambda spec: spec.system_id, kind="system")

    raw_analyses = raw_config.get("analyses")
    if not raw_analyses:
        raise ConfigError("'analyses' must be a non-empty list.")
    analyses = [parse_analysis_entry(raw_analysis, f"analyses[{position}]")
                for position, raw_analysis in enumerate(raw_analyses)]
    analyses = deduplicate(analyses, key=lambda spec: spec.analysis_id, kind="analysis")
    analysis_names = [spec.name for spec in analyses]
    if len(set(analysis_names)) != len(analysis_names):
        raise ConfigError("Analysis names must be unique.")

    limits = raw_config.get("limits", {})
    reject_unknown_keys(limits, LIMIT_KEYS, "limits")
    max_dimension = limits.get("max_dimension", 6000)
    if not isinstance(max_dimension, int) or max_dimension < 1:
        raise ConfigError("limits.max_dimension must be a positive integer.")

    return SweepConfig(
        sweep_name=str(raw_config.get("sweep_name", "sweep")),
        systems=tuple(systems),
        analyses=tuple(analyses),
        max_dimension=max_dimension,
        output_dir=raw_config.get("output_dir"),
        raw=raw_config,
    )


def reject_unknown_keys(entry, allowed_keys, context):
    if not isinstance(entry, dict):
        raise ConfigError(f"{context} must be a JSON object.")
    unknown_keys = set(entry) - allowed_keys
    if unknown_keys:
        raise ConfigError(f"{context}: unknown key(s) {sorted(unknown_keys)}; allowed: {sorted(allowed_keys)}")


def deduplicate(specs, key, kind):
    seen = {}
    for spec in specs:
        seen.setdefault(key(spec), spec)
    if len(seen) != len(specs):
        print(f"Note: {len(specs) - len(seen)} duplicate {kind} entr{'y' if len(specs) - len(seen) == 1 else 'ies'} ignored.")
    return list(seen.values())


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


# ---------------------------------------------------------------------------------------------
# Systems

def expand_geometries(entry, context):
    if "geometries" in entry and "geometry" in entry:
        raise ConfigError(f"{context}: give either 'geometries' or 'geometry', not both.")
    spec = entry.get("geometries", entry.get("geometry"))
    if spec is None:
        return [None]
    if isinstance(spec, dict):
        # Range forms: {"start", "stop", "num"} (inclusive linspace) or {"start", "stop", "step"}
        if set(spec) == {"start", "stop", "num"}:
            count = int(spec["num"])
            if count < 1:
                raise ConfigError(f"{context}: geometries.num must be >= 1.")
            if count == 1:
                return [float(spec["start"])]
            step = (spec["stop"] - spec["start"]) / (count - 1)
            return [round(spec["start"] + i * step, 10) for i in range(count)]
        if set(spec) == {"start", "stop", "step"}:
            if spec["step"] <= 0 or spec["stop"] < spec["start"]:
                raise ConfigError(f"{context}: geometries range needs step > 0 and stop >= start.")
            count = int(math.floor((spec["stop"] - spec["start"]) / spec["step"] + 1e-9)) + 1
            return [round(spec["start"] + i * spec["step"], 10) for i in range(count)]
        raise ConfigError(f"{context}: geometries must be a list, a number, {{start, stop, num}} or {{start, stop, step}}.")
    geometries = as_list(spec)
    for value in geometries:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{context}: geometry values must be numbers, got {value!r}.")
    return [float(value) for value in geometries]


def parse_active_space(raw_active_space, context):
    if raw_active_space is None:
        return None
    reject_unknown_keys(raw_active_space, {"active_electrons", "active_spatial_orbitals"}, f"{context}.active_space")
    active_electrons = raw_active_space.get("active_electrons")
    active_orbitals = raw_active_space.get("active_spatial_orbitals")
    if not (isinstance(active_electrons, int) and isinstance(active_orbitals, int)) or active_electrons < 1 or active_orbitals < 1:
        raise ConfigError(f"{context}.active_space needs positive integers active_electrons and active_spatial_orbitals.")
    if active_electrons > 2 * active_orbitals:
        raise ConfigError(f"{context}.active_space: {active_electrons} electrons do not fit in {active_orbitals} spatial orbitals.")
    return {"active_electrons": active_electrons, "active_spatial_orbitals": active_orbitals}


def parse_custom_atoms(raw_atoms, context):
    atoms = []
    for atom in raw_atoms:
        if (not isinstance(atom, (list, tuple)) or len(atom) != 2 or not isinstance(atom[0], str)
                or len(atom[1]) != 3):
            raise ConfigError(f"{context}.atoms entries must look like [\"H\", [x, y, z]].")
        atoms.append((atom[0], tuple(float(coordinate) for coordinate in atom[1])))
    return tuple(atoms)


def expand_system_entry(entry, context):
    molecule = entry.get("molecule")
    atoms = entry.get("atoms")
    if atoms is not None:
        if molecule not in (None, "custom"):
            raise ConfigError(f"{context}: 'atoms' defines a custom molecule; set molecule to \"custom\" or omit it.")
        if not entry.get("name"):
            raise ConfigError(f"{context}: custom molecules need a 'name'.")
        molecule = "custom"
        atoms = parse_custom_atoms(atoms, context)
    elif molecule not in LIBRARY_MOLECULES | GENERATED_MOLECULES:
        raise ConfigError(f"{context}: unknown molecule {molecule!r}; choose from "
                          f"{sorted(LIBRARY_MOLECULES | GENERATED_MOLECULES)} or give custom 'atoms'.")

    geometries = expand_geometries(entry, context)
    if molecule == "custom" and geometries != [None]:
        raise ConfigError(f"{context}: custom 'atoms' have a fixed geometry; remove 'geometries'.")
    if molecule != "custom" and geometries == [None]:
        raise ConfigError(f"{context}: {molecule} needs 'geometries'.")
    if molecule != "custom" and any(geometry <= 0 for geometry in geometries):
        raise ConfigError(f"{context}: geometry values (bond lengths in Angstrom, or the 'h4' angle) must be positive.")

    n_atoms = entry.get("n_atoms")
    if molecule == "h_chain":
        if not isinstance(n_atoms, int) or n_atoms < 2:
            raise ConfigError(f"{context}: h_chain needs an integer n_atoms >= 2.")
    elif n_atoms is not None:
        raise ConfigError(f"{context}: n_atoms only applies to molecule \"h_chain\".")

    backend = entry.get("backend", "pyscf")
    if backend not in BACKENDS:
        raise ConfigError(f"{context}: backend must be one of {sorted(BACKENDS)}.")
    if backend == "ham_gen" and (molecule == "custom" or entry.get("charge", 0) != 0):
        raise ConfigError(f"{context}: the ham_gen backend supports only neutral library molecules and h_chain.")

    charge = entry.get("charge", 0)
    if not isinstance(charge, int):
        raise ConfigError(f"{context}: charge must be an integer.")
    overlap_threshold = float(entry.get("overlap_threshold", 0.0))
    if overlap_threshold < 0:
        raise ConfigError(f"{context}: overlap_threshold must be >= 0.")

    bases = as_list(entry.get("basis", "sto-3g"))
    spins = as_list(entry.get("spin", "s0"))
    for spin in spins:
        if spin not in SPIN_LABELS:
            raise ConfigError(f"{context}: spin must be one of {sorted(SPIN_LABELS)}, got {spin!r}.")
    raw_active_spaces = entry["active_space"] if isinstance(entry.get("active_space"), list) else [entry.get("active_space")]
    active_spaces = [parse_active_space(raw, context) for raw in raw_active_spaces]

    systems = []
    for geometry, basis, spin, active_space in itertools.product(geometries, bases, spins, active_spaces):
        systems.append(SystemSpec(
            molecule=molecule, geometry=geometry, basis=str(basis), spin=spin, backend=backend,
            active_space=active_space, n_atoms=n_atoms, atoms=atoms, charge=charge,
            overlap_threshold=overlap_threshold, name=entry.get("name"),
        ))
    return systems


# ---------------------------------------------------------------------------------------------
# Analyses

def parse_target(raw_target, context):
    target = {"strategy": "lowest_with_overlap", "min_overlap": 1e-6}
    if raw_target is None:
        return target
    reject_unknown_keys(raw_target, {"strategy", "min_overlap", "index", "tolerance"}, f"{context}.target")
    target.update(raw_target)
    if target["strategy"] not in TARGET_STRATEGIES:
        raise ConfigError(f"{context}.target.strategy must be one of {sorted(TARGET_STRATEGIES)}.")
    if target["strategy"] == "index" and not isinstance(target.get("index"), int):
        raise ConfigError(f"{context}.target: strategy 'index' needs an integer 'index'.")
    if target["strategy"] == "lowest_in_sector":
        target.setdefault("tolerance", 1e-4)
    return target


def parse_analysis_entry(raw_analysis, context):
    reject_unknown_keys(raw_analysis, ANALYSIS_KEYS, context)
    name = raw_analysis.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError(f"{context}: every analysis needs a 'name'.")

    modes = as_list(raw_analysis.get("modes", ["exact"]))
    if not modes or any(mode not in BOUND_MODES for mode in modes):
        raise ConfigError(f"{context}: modes must be a non-empty subset of {sorted(BOUND_MODES)}. "
                          "('manual' mode needs hand-written intervals and is not sweepable.)")

    poly_degrees = raw_analysis.get("poly_degrees", [2, 4, 6, 8])
    poly_degrees = sorted({int(degree) for degree in as_list(poly_degrees)})
    if not poly_degrees or poly_degrees[0] < 0 or poly_degrees[-1] > MAX_MOMENT_DEGREE:
        raise ConfigError(f"{context}: poly_degrees must lie in [0, {MAX_MOMENT_DEGREE}] (moments stored by calc_eigen).")

    bound_extension = raw_analysis.get("bound_extension")
    # bound_extension is a fraction of the gap to each neighbouring level (gen_bound_pairs requires 0 < x < 1/2)
    if "approx" in modes and (not isinstance(bound_extension, (int, float)) or not 0 < bound_extension < 0.5):
        raise ConfigError(f"{context}: approx mode needs 'bound_extension' between 0 and 0.5 "
                          "(a fraction of the energy gap to each neighbouring level).")
    energy_threshold = raw_analysis.get("energy_threshold")
    if "below" in modes and not (energy_threshold == "HF" or isinstance(energy_threshold, (int, float))):
        raise ConfigError(f"{context}: below mode needs 'energy_threshold' (a number in Hartree, or \"HF\").")

    recipe_overrides = raw_analysis.get("recipe_overrides", {})
    if not isinstance(recipe_overrides, dict):
        raise ConfigError(f"{context}: recipe_overrides must be an object.")
    reserved = {"mode", "eig_results", "target_indices", "b_values", "poly_degrees", "poly_degree"}
    if reserved & set(recipe_overrides):
        raise ConfigError(f"{context}: recipe_overrides cannot set {sorted(reserved & set(recipe_overrides))}; "
                          "those are filled in by the sweep.")

    return AnalysisSpec(
        name=name,
        target=parse_target(raw_analysis.get("target"), context),
        modes=tuple(modes),
        poly_degrees=tuple(poly_degrees),
        bound_extension=bound_extension,
        energy_threshold=energy_threshold,
        solver=str(raw_analysis.get("solver", "SCS")),
        save_plots=bool(raw_analysis.get("save_plots", False)),
        recipe_overrides=recipe_overrides,
    )
