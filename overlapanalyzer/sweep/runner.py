"""
Sweep orchestration: planning, per-system execution with stage caching, failure isolation, resume, summaries.

Unit of work: one system together with every analysis that still needs computing for it. Within that unit the
stages run in order (Hamiltonian -> exact diagonalization -> per-analysis spectrum summary and bounds), and each
completed stage is recorded on disk, so an interrupted sweep restarts from the first unfinished stage.

Cache validity is tracked explicitly:
- each stage record carries a version (bumped when the code producing it changes) and a fresh token;
- the eigen stage stores the Hamiltonian token it was built from, and each result stores the eigen token,
  so recomputing an upstream stage (e.g. with --force) invalidates everything derived from it.

Resume rules for an experiment (system, analysis):
- current result marked "complete"                   -> skipped (unless --force)
- current result marked "failed"                     -> skipped (unless --retry-failed or --force)
- missing, interrupted, or stale (older code/inputs) -> computed, reusing any valid system stages
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import os
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, asdict

from overlapanalyzer.sweep.bounds import compute_all_bounds
from overlapanalyzer.sweep.hamiltonian import HamiltonianInfo, generate_hamiltonian
from overlapanalyzer.sweep.spectrum import (eigen_json_path, run_exact_diagonalization, summarize_spectrum,
                                            truncated_index_of)
from overlapanalyzer.sweep.storage import (OutputDirectoryLock, SystemPaths, StageStatus, capture_output,
                                           collect_provenance, read_json, utc_timestamp, write_json_atomic)

# Bump a version when the code producing that stage's output changes meaningfully; cached outputs from older
# versions are then recomputed. eigen v2: calc_eigen's 's1' reference corrected from triplet to singlet.
STAGE_VERSIONS = {"hamiltonian": 1, "eigen": 2}
# Bump when the content of result.json changes (target selection, bounds parsing, validation).
# v3: approx-mode bounds for targets above merged non-target regions corrected (process_bound_pairs).
ANALYSIS_VERSION = 3

# BLAS/OpenMP thread variables limited per worker in parallel runs (otherwise every worker uses every core)
THREAD_ENVIRONMENT_VARIABLES = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]


@dataclass
class ExperimentOutcome:
    system_id: str
    analysis_id: str
    label: str
    state: str          # complete | failed | cached-complete | cached-failed | not-run
    message: str = ""
    seconds: float = 0.0


@dataclass
class SystemWork:
    system: object
    analyses: list      # AnalysisSpec objects that need computing
    skipped: list       # ExperimentOutcome for analyses already done (or failed, without retry)


# ---------------------------------------------------------------------------------------------
# Cache validity

def current_hamiltonian_token(stage_status: StageStatus) -> str | None:
    record = stage_status.get("hamiltonian")
    hamiltonian_file = (record.get("info") or {}).get("hamiltonian_file")
    if hamiltonian_file and stage_status.is_complete("hamiltonian", hamiltonian_file, STAGE_VERSIONS["hamiltonian"]):
        return stage_status.token("hamiltonian")
    return None


def current_eigen_token(stage_status: StageStatus) -> str | None:
    hamiltonian_token = current_hamiltonian_token(stage_status)
    if hamiltonian_token is None:
        return None
    eigen_file = stage_status.get("eigen").get("eigen_file")
    if eigen_file and stage_status.is_complete("eigen", eigen_file, STAGE_VERSIONS["eigen"],
                                               built_from=("hamiltonian_token", hamiltonian_token)):
        return stage_status.token("eigen")
    return None


def read_experiment_state(output_dir: str, system, analysis) -> str:
    """complete | failed | stale | pending, taking code versions and upstream recomputation into account."""
    paths = SystemPaths(output_dir, system.system_id)
    result = read_json(paths.result_json(analysis.analysis_id))
    state = (result or {}).get("status")
    if state not in ("complete", "failed"):
        return "pending"   # Missing, unreadable, or left 'running' by an interruption
    if result.get("analysis_version") != ANALYSIS_VERSION:
        return "stale"
    built_from_eigen = result.get("eigen_token")
    if built_from_eigen is not None and built_from_eigen != current_eigen_token(StageStatus(paths.status_json)):
        return "stale"
    return state


# ---------------------------------------------------------------------------------------------
# Planning

def experiment_label(system, analysis) -> str:
    return f"{system.label} :: {analysis.name}"


def matches_filter(system, analysis, only: str | None) -> bool:
    if not only:
        return True
    return only in system.system_id or only in analysis.name


def plan_sweep(config, output_dir: str, retry_failed: bool = False, force: bool = False, only: str | None = None):
    work_items = []
    for system in config.systems:
        needed, skipped = [], []
        for analysis in config.analyses:
            if not matches_filter(system, analysis, only):
                continue
            state = read_experiment_state(output_dir, system, analysis)
            label = experiment_label(system, analysis)
            if force or state in ("pending", "stale") or (state == "failed" and retry_failed):
                needed.append(analysis)
            else:
                skipped.append(ExperimentOutcome(system.system_id, analysis.analysis_id, label, f"cached-{state}"))
        if needed or skipped:
            work_items.append(SystemWork(system, needed, skipped))
    return work_items


# ---------------------------------------------------------------------------------------------
# Execution of one system

def run_stage(stage_status: StageStatus, stage: str, log_path: str, compute):
    """Run one system stage with timing, log capture and an on-disk record of its outcome."""
    stage_status.record(stage, "running", version=STAGE_VERSIONS[stage])
    start_time = time.perf_counter()
    try:
        with capture_output(log_path):
            result = compute()
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(traceback.format_exc())
        stage_status.record(stage, "failed", version=STAGE_VERSIONS[stage], seconds=elapsed,
                            error=f"{type(exc).__name__}: {exc}")
        raise
    elapsed = time.perf_counter() - start_time
    return result, elapsed


def experiment_record(system, analysis) -> dict:
    return {
        "system_id": system.system_id,
        "analysis_id": analysis.analysis_id,
        "system": {**system.identity(), "name": system.name, "label": system.label},
        "analysis": asdict(analysis),
    }


def write_failed_results(paths: SystemPaths, system, analyses, error: str, timing: dict, outcomes: list, report):
    for analysis in analyses:
        write_json_atomic(paths.result_json(analysis.analysis_id), {
            "status": "failed",
            "analysis_version": ANALYSIS_VERSION,
            "error": error,
            "experiment": experiment_record(system, analysis),
            "timing": timing,
            "finished": utc_timestamp(),
        })
        outcome = ExperimentOutcome(system.system_id, analysis.analysis_id, experiment_label(system, analysis),
                                    "failed", error)
        outcomes.append(outcome)
        report(outcome)


def ensure_hamiltonian(system, paths: SystemPaths, stage_status: StageStatus, force: bool):
    if not force and current_hamiltonian_token(stage_status):
        record = stage_status.get("hamiltonian")
        return HamiltonianInfo.from_dict(record["info"]), {"seconds": record.get("seconds"), "cached": True}

    info, elapsed = run_stage(stage_status, "hamiltonian", paths.log, lambda: generate_hamiltonian(system, paths.root))
    stage_status.record("hamiltonian", "complete", version=STAGE_VERSIONS["hamiltonian"], token=uuid.uuid4().hex,
                        seconds=elapsed, info=info.to_dict())
    write_json_atomic(paths.system_json, {"system": {**system.identity(), "name": system.name, "label": system.label},
                                          "hamiltonian": info.to_dict()})
    return info, {"seconds": elapsed, "cached": False}


def ensure_spectrum(system, info, paths: SystemPaths, stage_status: StageStatus, max_dimension: int, force: bool):
    if not force and current_eigen_token(stage_status):
        record = stage_status.get("eigen")
        return record["eigen_file"], {"seconds": record.get("seconds"), "cached": True}

    eigen_path = eigen_json_path(paths.root, info, system.spin)
    _, elapsed = run_stage(stage_status, "eigen", paths.log,
                           lambda: run_exact_diagonalization(system, info, paths.root, max_dimension))
    stage_status.record("eigen", "complete", version=STAGE_VERSIONS["eigen"], token=uuid.uuid4().hex,
                        hamiltonian_token=stage_status.token("hamiltonian"), seconds=elapsed, eigen_file=eigen_path)
    return eigen_path, {"seconds": elapsed, "cached": False}


def load_spectrum(system, info, paths: SystemPaths, stage_status: StageStatus, max_dimension: int, timing: dict,
                  force: bool):
    """Load (computing if needed) the diagonalization; a corrupt or unreadable cached file is recomputed once."""
    eigen_path, timing["eigen"] = ensure_spectrum(system, info, paths, stage_status, max_dimension, force)
    try:
        with open(eigen_path, "r", encoding="utf-8") as eigen_file:
            return json.load(eigen_file)
    except (OSError, json.JSONDecodeError):
        if not timing["eigen"]["cached"]:
            raise   # Freshly written and still unreadable: a genuine failure
    stage_status.record("eigen", "invalid", version=STAGE_VERSIONS["eigen"], error="cached eigen file unreadable")
    eigen_path, timing["eigen"] = ensure_spectrum(system, info, paths, stage_status, max_dimension, force=True)
    with open(eigen_path, "r", encoding="utf-8") as eigen_file:
        return json.load(eigen_file)


def collect_warnings(result: dict) -> list:
    """Scientific red flags worth a human look; they do not make an experiment fail."""
    warnings = []
    validation = (result.get("exact") or {}).get("validation") or {}
    if validation and not validation.get("moments_consistent", True):
        warnings.append(f"spectrum/moment mismatch {validation['moment_consistency_max_error']:.2e}")
    if validation and not validation.get("reference_spin_consistent", True):
        warnings.append(f"reference <S^2> differs from its spin label by {validation['reference_s2_error']:.3g}")
    for mode, mode_record in (result.get("bounds") or {}).items():
        if mode_record.get("status") in ("partial", "no_solution"):
            warnings.append(f"{mode}: solver returned no bound for some degrees ({mode_record['status']})")
        for row in mode_record.get("degrees") or []:
            if row.get("brackets_exact") is False:
                warnings.append(f"{mode} degree {row['degree']}: bounds do not contain the exact value")
            for side, status in (row.get("solver_status") or {}).items():
                if status != "optimal":
                    warnings.append(f"{mode} degree {row['degree']} {side}: solver status {status}")
    return warnings


def run_analysis(system, analysis, info, eig_results, eigen_token, paths: SystemPaths, timing: dict) -> ExperimentOutcome:
    analysis_dir = paths.analysis_dir(analysis.analysis_id)
    os.makedirs(analysis_dir, exist_ok=True)
    start_time = time.perf_counter()
    result = {
        "status": "running",
        "analysis_version": ANALYSIS_VERSION,
        "eigen_token": eigen_token,
        "experiment": experiment_record(system, analysis),
        "system": info.to_dict(),
    }
    error = None
    try:
        # Exact part: which level is the target, its energy/overlap/spin, gaps, self-consistency checks
        result["exact"] = summarize_spectrum(eig_results, analysis.target, system.spin)
        target = result["exact"]["target"]
        target_truncated_index = truncated_index_of(eig_results, target["energy"])
        if target_truncated_index is None:
            raise RuntimeError(f"Target level has overlap {target['overlap']:.3g}, below the system's "
                               f"overlap_threshold {system.overlap_threshold}; it is not a bound constraint level.")
        # Estimated part: polynomial bounds from moments
        with capture_output(os.path.join(analysis_dir, "analysis.log")):
            result["bounds"] = compute_all_bounds(analysis, eig_results, target_truncated_index,
                                                  target["overlap"], analysis_dir)
        # A mode 'failed' only on an exception (worth retrying); 'partial'/'no_solution' are solver outcomes
        failed_modes = [mode for mode, record in result["bounds"].items() if record["status"] == "failed"]
        if failed_modes:
            error = "bound mode(s) failed: " + "; ".join(f"{mode}: {result['bounds'][mode]['error']}" for mode in failed_modes)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with open(os.path.join(analysis_dir, "analysis.log"), "a", encoding="utf-8") as log_file:
            log_file.write(traceback.format_exc())

    elapsed = time.perf_counter() - start_time
    result["status"] = "failed" if error else "complete"
    result["error"] = error
    result["warnings"] = collect_warnings(result)
    result["timing"] = {**timing, "analysis_seconds": elapsed}
    result["finished"] = utc_timestamp()
    write_json_atomic(paths.result_json(analysis.analysis_id), result)
    message = error or ("; ".join(result["warnings"][:2]) if result["warnings"] else "")
    return ExperimentOutcome(system.system_id, analysis.analysis_id, experiment_label(system, analysis),
                             result["status"], message, elapsed)


def process_system(work: SystemWork, output_dir: str, max_dimension: int, force: bool, report=lambda outcome: None):
    """Compute everything still needed for one system. Never raises for scientific failures; only interrupts propagate."""
    system = work.system
    outcomes = list(work.skipped)
    for outcome in work.skipped:
        report(outcome)
    if not work.analyses:
        return outcomes

    paths = SystemPaths(output_dir, system.system_id)
    os.makedirs(paths.root, exist_ok=True)
    stage_status = StageStatus(paths.status_json)
    timing = {}
    try:
        info, timing["hamiltonian"] = ensure_hamiltonian(system, paths, stage_status, force)
        eig_results = load_spectrum(system, info, paths, stage_status, max_dimension, timing, force)
    except Exception as exc:
        failed_stage = "eigen" if "hamiltonian" in timing else "hamiltonian"
        write_failed_results(paths, system, work.analyses, f"{failed_stage} stage: {type(exc).__name__}: {exc}",
                             timing, outcomes, report)
        return outcomes

    eigen_token = stage_status.token("eigen")
    # Attribute freshly computed (non-cached) system stage time to the first analysis, so reported times add up
    fresh_stage_seconds = sum(stage["seconds"] or 0.0 for stage in timing.values() if not stage["cached"])
    for analysis in work.analyses:
        outcome = run_analysis(system, analysis, info, eig_results, eigen_token, paths, timing)
        outcome.seconds += fresh_stage_seconds
        fresh_stage_seconds = 0.0
        outcomes.append(outcome)
        report(outcome)
    return outcomes


# ---------------------------------------------------------------------------------------------
# Whole sweep

def record_invocation(config, output_dir: str, options: dict) -> None:
    sweep_path = os.path.join(output_dir, "sweep.json")
    sweep_record = read_json(sweep_path, default={}) or {}
    sweep_record["sweep_name"] = config.sweep_name
    sweep_record["config"] = config.raw
    sweep_record["experiment_count"] = config.experiment_count
    sweep_record.setdefault("invocations", []).append({**collect_provenance(), "options": options})
    write_json_atomic(sweep_path, sweep_record)


def run_sweep(config, output_dir: str, jobs: int = 1, retry_failed: bool = False, force: bool = False,
              only: str | None = None, report=lambda outcome: None) -> list:
    os.makedirs(output_dir, exist_ok=True)
    with OutputDirectoryLock(output_dir):
        record_invocation(config, output_dir, {"jobs": jobs, "retry_failed": retry_failed, "force": force, "only": only})
        work_items = plan_sweep(config, output_dir, retry_failed=retry_failed, force=force, only=only)
        outcomes = []
        try:
            if jobs <= 1:
                for work in work_items:
                    outcomes.extend(process_system(work, output_dir, config.max_dimension, force, report))
            else:
                outcomes.extend(run_in_parallel(work_items, output_dir, config.max_dimension, force, jobs, report))
        finally:
            # Summaries reflect everything on disk, including work finished before an interruption
            write_summaries(config, output_dir)
    return outcomes


def recover_crashed_system(work: SystemWork, output_dir: str, error: Exception) -> list:
    """
    After a worker died, keep results it finished. A system that was mid-stage when the pool broke is recorded
    as failed (it may be what crashed, e.g. out of memory; --retry-failed reruns it). A system that never started
    is left pending, so the next run simply computes it.
    """
    paths = SystemPaths(output_dir, work.system.system_id)
    stage_records = StageStatus(paths.status_json).stages
    system_was_running = any(record.get("state") == "running" for record in stage_records.values())
    outcomes, unfinished = [], []
    for analysis in work.analyses:
        label = experiment_label(work.system, analysis)
        if read_experiment_state(output_dir, work.system, analysis) == "complete":
            outcomes.append(ExperimentOutcome(work.system.system_id, analysis.analysis_id, label, "complete",
                                              "finished before the worker failed"))
        elif system_was_running:
            unfinished.append(analysis)
        else:
            outcomes.append(ExperimentOutcome(work.system.system_id, analysis.analysis_id, label, "not-run",
                                              "not started: the worker pool crashed; rerun to compute"))
    write_failed_results(paths, work.system, unfinished, f"worker process failed: {type(error).__name__}: {error}",
                         {}, outcomes, lambda outcome: None)
    return outcomes


class limited_worker_threads:
    """Give each of `jobs` workers an equal share of cores, unless the user already set thread limits."""

    def __init__(self, jobs: int):
        self.threads = str(max(1, (os.cpu_count() or 1) // jobs))
        self.previous = {}

    def __enter__(self):
        if not any(name in os.environ for name in THREAD_ENVIRONMENT_VARIABLES):
            for name in THREAD_ENVIRONMENT_VARIABLES:
                self.previous[name] = os.environ.get(name)
                os.environ[name] = self.threads
        return self

    def __exit__(self, *exc_info):
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def stop_workers(executor: ProcessPoolExecutor) -> None:
    """Cancel queued systems and stop running workers (their partial stages are redone on resume)."""
    worker_processes = list((getattr(executor, "_processes", None) or {}).values())   # shutdown() clears this table
    executor.shutdown(wait=False, cancel_futures=True)
    for process in worker_processes:
        if process.is_alive():
            process.terminate()
    for process in worker_processes:
        process.join(timeout=10)


def run_in_parallel(work_items, output_dir, max_dimension, force, jobs, report):
    outcomes = []
    for work in work_items:     # Cached experiments need no worker
        for outcome in work.skipped:
            outcomes.append(outcome)
            report(outcome)
    pending_work = [SystemWork(work.system, work.analyses, []) for work in work_items if work.analyses]
    if not pending_work:
        return outcomes

    # 'spawn' gives each worker a clean interpreter; workers inherit the thread limits from the environment
    with limited_worker_threads(jobs):
        executor = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"))
        try:
            future_to_work = {executor.submit(process_system, work, output_dir, max_dimension, force): work
                              for work in pending_work}
            for future in as_completed(future_to_work):
                work = future_to_work[future]
                try:
                    system_outcomes = future.result()
                except (BrokenProcessPool, Exception) as exc:
                    # A worker died (e.g. out of memory): the sweep still finishes and can be resumed later
                    system_outcomes = recover_crashed_system(work, output_dir, exc)
                for outcome in system_outcomes:
                    outcomes.append(outcome)
                    report(outcome)
        except BaseException:
            stop_workers(executor)
            raise
        executor.shutdown(wait=True)
    return outcomes


# ---------------------------------------------------------------------------------------------
# Summaries

def deepest_degree(mode_record: dict):
    degrees = mode_record.get("degrees") or []
    return degrees[-1] if degrees else None


def flatten_result(system, analysis, result: dict | None, state: str) -> tuple[dict, list]:
    """One summary row for the experiment, plus one row per (mode, degree) bound."""
    row = {
        "system_id": system.system_id, "analysis_id": analysis.analysis_id, "analysis": analysis.name,
        "molecule": system.name or system.molecule, "geometry": system.geometry, "basis": system.basis,
        "spin": system.spin, "backend": system.backend,
        "active_space": (f"{system.active_space['active_electrons']}e{system.active_space['active_spatial_orbitals']}o"
                         if system.active_space else ""),
        "status": state,
    }
    bound_rows = []
    if not result or state in ("pending", "stale"):
        return row, bound_rows

    system_info = result.get("system", {})
    exact = result.get("exact", {})
    target = exact.get("target", {})
    ground = exact.get("global_ground_state", {})
    validation = exact.get("validation", {})
    timing = result.get("timing", {})
    warnings = result.get("warnings") or []
    row.update({
        "n_electrons": system_info.get("n_electrons"),
        "n_spin_orbitals": system_info.get("n_spin_orbitals"),
        "hilbert_dimension": system_info.get("hilbert_dimension"),
        "hf_energy": system_info.get("hf_energy"),
        "exact_target_strategy": target.get("strategy"),
        "exact_target_index": target.get("index"),
        "exact_target_energy": target.get("energy"),
        "exact_target_overlap": target.get("overlap"),
        "exact_target_s2": target.get("s2"),
        "exact_target_is_global_ground": target.get("is_global_ground_state"),
        "exact_target_gap_above_visible": target.get("gap_above_visible"),
        "exact_ground_energy": ground.get("energy"),
        "exact_ground_overlap": ground.get("overlap"),
        "check_moments_consistent": validation.get("moments_consistent"),
        "check_reference_spin_consistent": validation.get("reference_spin_consistent"),
        "hamiltonian_seconds": (timing.get("hamiltonian") or {}).get("seconds"),
        "eigen_seconds": (timing.get("eigen") or {}).get("seconds"),
        "analysis_seconds": timing.get("analysis_seconds"),
        "n_warnings": len(warnings),
        "warnings": "; ".join(warnings),
        "error": result.get("error") or "",
    })
    for mode, mode_record in (result.get("bounds") or {}).items():
        deepest = deepest_degree(mode_record)
        if deepest:
            row.update({f"bound_{mode}_degree": deepest["degree"], f"bound_{mode}_lower": deepest["lower"],
                        f"bound_{mode}_upper": deepest["upper"], f"bound_{mode}_brackets_exact": deepest.get("brackets_exact")})
        for degree_row in mode_record.get("degrees") or []:
            bound_rows.append({
                "system_id": system.system_id, "analysis": analysis.name, "molecule": row["molecule"],
                "geometry": system.geometry, "basis": system.basis, "spin": system.spin, "mode": mode,
                "degree": degree_row["degree"], "lower": degree_row["lower"], "upper": degree_row["upper"],
                "width": degree_row["width"], "exact_reference_value": mode_record.get("exact_reference_value"),
                "brackets_exact": degree_row.get("brackets_exact"),
                "solver_status": ";".join(f"{side}={status}" for side, status in (degree_row.get("solver_status") or {}).items()),
            })
    return row, bound_rows


def write_csv(path: str, rows: list) -> None:
    fieldnames = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def write_summaries(config, output_dir: str) -> dict:
    experiment_rows, bound_rows = [], []
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    with open(jsonl_path + ".tmp", "w", encoding="utf-8") as jsonl_file:
        for system in config.systems:
            for analysis in config.analyses:
                result = read_json(SystemPaths(output_dir, system.system_id).result_json(analysis.analysis_id))
                state = read_experiment_state(output_dir, system, analysis)
                row, rows_for_bounds = flatten_result(system, analysis, result, state)
                experiment_rows.append(row)
                bound_rows.extend(rows_for_bounds)
                record = result if result and state not in ("pending", "stale") else {**row}
                jsonl_file.write(json.dumps(record) + "\n")
    os.replace(jsonl_path + ".tmp", jsonl_path)
    write_csv(os.path.join(output_dir, "results.csv"), experiment_rows)
    write_csv(os.path.join(output_dir, "bounds.csv"), bound_rows)

    counts = {}
    for row in experiment_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts
