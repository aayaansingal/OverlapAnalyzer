"""
On-disk layout, crash-safe JSON writes, stage bookkeeping and provenance for sweeps.

Layout of an output directory:

    <output>/
        sweep.json                     resolved config, provenance, invocation history
        results.csv / results.jsonl    one row per (system, analysis) experiment
        bounds.csv                     one row per (experiment, mode, polynomial degree)
        systems/<system_id>/
            system.json                the SystemSpec and Hamiltonian facts (electrons, orbitals, atoms, HF energy)
            status.json                per-stage state: hamiltonian, eigen
            system.log                 captured output of the library calls
            ham_fer/, info.csv, eigen/ files written by the library, in the layout eigen.calc_eigen expects
            analyses/<analysis_id>/
                result.json            exact quantities, bounds, timings, status
                analysis.log
                bounds_<mode>.json     distilled OverlapEstimator payload (coefficients included)
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata

import numpy as np

TRACKED_PACKAGES = ["numpy", "scipy", "openfermion", "openfermionpyscf", "pyscf", "cvxpy", "Mosek", "overlapanalyzer"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(value):
    """Convert numpy scalars/arrays and tuples into plain JSON types (NaN/inf become None)."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, complex):
        return to_jsonable(value.real) if abs(value.imag) < 1e-12 else [value.real, value.imag]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json_atomic(path: str, payload) -> None:
    """Write JSON via a temporary file and rename, so an interrupted run never leaves a truncated file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2)
        os.replace(temporary_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)
        raise


def read_json(path: str, default=None):
    """Read JSON, treating a missing or unreadable file as absent (resume then recomputes)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class SystemPaths:
    def __init__(self, output_dir: str, system_id: str):
        self.root = os.path.join(output_dir, "systems", system_id)
        self.system_json = os.path.join(self.root, "system.json")
        self.status_json = os.path.join(self.root, "status.json")
        self.log = os.path.join(self.root, "system.log")

    def analysis_dir(self, analysis_id: str) -> str:
        return os.path.join(self.root, "analyses", analysis_id)

    def result_json(self, analysis_id: str) -> str:
        return os.path.join(self.analysis_dir(analysis_id), "result.json")


class StageStatus:
    """Per-system stage records. A stage counts as done only if it was marked complete *and* its output still exists."""

    def __init__(self, path: str):
        self.path = path
        self.stages = read_json(path, default={}) or {}

    def is_complete(self, stage: str, required_file: str | None = None) -> bool:
        record = self.stages.get(stage, {})
        if record.get("state") != "complete":
            return False
        return required_file is None or os.path.exists(required_file)

    def record(self, stage: str, state: str, **details) -> None:
        self.stages[stage] = {"state": state, "updated": utc_timestamp(), **details}
        write_json_atomic(self.path, self.stages)

    def get(self, stage: str) -> dict:
        return self.stages.get(stage, {})


@contextlib.contextmanager
def capture_output(log_path: str):
    """Send stdout/stderr of chatty library code (calc_eigen, PySCF, cvxpy) to a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {utc_timestamp()} =====\n")
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            yield log_file


def collect_provenance() -> dict:
    """Code version and package versions, so results can be traced to the software that produced them."""
    repository_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    git_info = {}
    try:
        git_info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True,
                                            text=True, check=True, timeout=10).stdout.strip()
        dirty_output = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_root,
                                      capture_output=True, text=True, check=True, timeout=10).stdout
        git_info["uncommitted_changes"] = bool(dirty_output.strip())
    except (OSError, subprocess.SubprocessError):
        git_info["commit"] = None

    package_versions = {}
    for package in TRACKED_PACKAGES:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None

    return {
        "timestamp": utc_timestamp(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_info,
        "packages": package_versions,
    }
