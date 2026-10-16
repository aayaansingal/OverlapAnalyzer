"""
Reliability of the sweep runner: cache invalidation, corruption recovery, locking, worker crashes,
parallel determinism, interruption/resume through the real CLI, and graceful failure of optional backends.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

pytest.importorskip("openfermionpyscf")

from overlapanalyzer.sweep import cli, runner
from overlapanalyzer.sweep.config import parse_sweep_config
from overlapanalyzer.sweep.runner import (SystemWork, read_experiment_state, recover_crashed_system, run_sweep,
                                          limited_worker_threads, THREAD_ENVIRONMENT_VARIABLES)
from overlapanalyzer.sweep.storage import OutputDirectoryLock, StageStatus, SweepLockedError, SystemPaths

from conftest import load_result

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TWO_ANALYSES = {
    "systems": [{"molecule": "h2", "geometries": [0.74]}],
    "analyses": [{"name": "first", "modes": ["exact"], "poly_degrees": [2, 4]},
                 {"name": "second", "target": {"strategy": "max_overlap"}, "modes": ["exact"], "poly_degrees": [2]}],
}


def bounds_table(output_dir):
    table = pd.read_csv(os.path.join(output_dir, "bounds.csv"))
    return table.sort_values(["system_id", "analysis", "mode", "degree"]).reset_index(drop=True)


OPTIMAL = "lower=optimal;upper=optimal"


def assert_same_bounds(expected, actual):
    """
    Two runs of the same sweep must agree: exactly for LP ('exact' mode) results, and within SCS's
    tolerance for SDP results the solver reports as optimal. Non-optimal SDP solves can drift between
    runs on multithreaded BLAS (they are flagged in bounds.csv), so only their presence is compared.
    """
    keys = ["system_id", "analysis", "mode", "degree"]
    merged = expected.merge(actual, on=keys, suffixes=("_expected", "_actual"))
    assert len(merged) == len(expected) == len(actual)
    difference = (merged[["lower_expected", "upper_expected"]].to_numpy()
                  - merged[["lower_actual", "upper_actual"]].to_numpy())
    merged["difference"] = abs(difference).max(axis=1)
    lp_rows = merged["mode"] == "exact"
    optimal_sdp_rows = ~lp_rows & (merged["solver_status_expected"] == OPTIMAL) & (merged["solver_status_actual"] == OPTIMAL)

    def one_line_report(rows, limit):
        # Single line, so CI's failure annotation shows the offending rows
        offending = merged[rows & (merged["difference"] >= limit)]
        return " | ".join(f"{row.system_id[:20]} {row.mode} deg{row.degree} diff={row.difference:.3g} "
                          f"status {row.solver_status_expected} vs {row.solver_status_actual}"
                          for row in offending.itertuples())

    assert merged.loc[lp_rows, "difference"].max() < 1e-9, "LP rows differ: " + one_line_report(lp_rows, 1e-9)
    if optimal_sdp_rows.any():
        assert merged.loc[optimal_sdp_rows, "difference"].max() < 1e-3, \
            "optimal SDP rows differ: " + one_line_report(optimal_sdp_rows, 1e-3)


# --- Cache validity -------------------------------------------------------------------------------

def test_forced_recompute_marks_dependent_analyses_stale(run_config):
    config, output_dir, _ = run_config(TWO_ANALYSES)
    system, first, second = config.systems[0], *config.analyses
    original = load_result(output_dir, system, second)

    run_sweep(config, output_dir, force=True, only="first")   # Recomputes the system's Hamiltonian and spectrum
    assert read_experiment_state(output_dir, system, first) == "complete"
    assert read_experiment_state(output_dir, system, second) == "stale"   # built from the old diagonalization

    outcomes = run_sweep(config, output_dir)
    assert {o.analysis_id: o.state for o in outcomes}[second.analysis_id] == "complete"
    refreshed = load_result(output_dir, system, second)
    assert refreshed["eigen_token"] != original["eigen_token"]
    assert refreshed["exact"]["target"]["energy"] == pytest.approx(original["exact"]["target"]["energy"], abs=1e-12)


def test_code_version_bumps_invalidate_cached_work(run_config, monkeypatch):
    config, output_dir, _ = run_config(TWO_ANALYSES)
    system, first = config.systems[0], config.analyses[0]

    monkeypatch.setattr(runner, "ANALYSIS_VERSION", runner.ANALYSIS_VERSION + 1)
    assert read_experiment_state(output_dir, system, first) == "stale"

    monkeypatch.setattr(runner, "STAGE_VERSIONS", {**runner.STAGE_VERSIONS, "eigen": runner.STAGE_VERSIONS["eigen"] + 1})
    run_sweep(config, output_dir)
    result = load_result(output_dir, system, first)
    assert result["status"] == "complete"
    assert result["timing"]["hamiltonian"]["cached"] and not result["timing"]["eigen"]["cached"]


def test_corrupt_cached_spectrum_is_recomputed(run_config):
    config, output_dir, _ = run_config(TWO_ANALYSES)
    eigen_file = StageStatus(SystemPaths(output_dir, config.systems[0].system_id).status_json).get("eigen")["eigen_file"]
    with open(eigen_file, "w") as handle:
        handle.write('{"truncated": ')                     # simulate a damaged file

    extended = dict(TWO_ANALYSES, analyses=TWO_ANALYSES["analyses"] + [
        {"name": "third", "target": {"strategy": "index", "index": 0}, "modes": ["exact"], "poly_degrees": [2]}])
    config, output_dir, _ = run_config(extended, output_dir=output_dir)
    result = load_result(output_dir, config.systems[0], config.analyses[2])
    assert result["status"] == "complete" and not result["timing"]["eigen"]["cached"]


def test_pyscf_scratch_file_is_removed(run_config):
    config, output_dir, _ = run_config(TWO_ANALYSES)
    system_dir = SystemPaths(output_dir, config.systems[0].system_id).root
    assert not [name for name in os.listdir(system_dir) if name.endswith(".hdf5")]


# --- Locking and crash recovery ------------------------------------------------------------------------

def test_output_directory_lock(tmp_path):
    with OutputDirectoryLock(str(tmp_path)):
        with pytest.raises(SweepLockedError):
            OutputDirectoryLock(str(tmp_path)).acquire()
        with pytest.raises(SweepLockedError):
            run_sweep(parse_sweep_config(TWO_ANALYSES), str(tmp_path))
    # A lock left by a process that no longer exists is taken over
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    lock_path = os.path.join(str(tmp_path), ".sweep.lock")
    with open(lock_path, "w") as handle:
        json.dump({"pid": finished.pid, "host": __import__("socket").gethostname()}, handle)
    with OutputDirectoryLock(str(tmp_path)):
        pass
    assert not os.path.exists(lock_path)


def test_crash_recovery_keeps_finished_results(run_config):
    config, output_dir, _ = run_config(TWO_ANALYSES)
    system, first, second = config.systems[0], *config.analyses
    paths = SystemPaths(output_dir, system.system_id)
    os.remove(paths.result_json(second.analysis_id))   # 'second' never finished
    work = SystemWork(system, [first, second], [])

    # The system was not mid-stage when the pool broke: unfinished work stays pending, nothing is marked failed
    outcomes = recover_crashed_system(work, output_dir, MemoryError("worker killed"))
    assert {o.analysis_id: o.state for o in outcomes} == {first.analysis_id: "complete", second.analysis_id: "not-run"}
    assert read_experiment_state(output_dir, system, second) == "pending"

    # The system was mid-stage (possibly the cause of the crash), so it has no current results yet:
    # its experiments are recorded as retryable failures
    StageStatus(paths.status_json).record("eigen", "running")
    outcomes = recover_crashed_system(work, output_dir, MemoryError("worker killed"))
    assert {o.state for o in outcomes} == {"failed"}
    assert "worker killed" in load_result(output_dir, system, second)["error"]
    assert read_experiment_state(output_dir, system, second) == "failed"


# --- Parallel execution -----------------------------------------------------------------------------

PARALLEL_CONFIG = {
    "systems": [{"molecule": "h2", "geometries": [0.6, 0.9, 1.4]}],
    "analyses": [{"name": "ground", "modes": ["exact", "approx"], "poly_degrees": [2, 4], "bound_extension": 0.05}],
}


def test_parallel_matches_sequential(run_config, tmp_path):
    _, sequential_dir, _ = run_config(PARALLEL_CONFIG, output_dir=tmp_path / "sequential")
    _, parallel_dir, outcomes = run_config(PARALLEL_CONFIG, output_dir=tmp_path / "parallel", jobs=2)
    assert {outcome.state for outcome in outcomes} == {"complete"}
    sequential, parallel = bounds_table(sequential_dir), bounds_table(parallel_dir)
    assert len(sequential) == len(parallel) == 12
    assert_same_bounds(sequential, parallel)


def test_worker_thread_limits_are_scoped(monkeypatch):
    for name in THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with limited_worker_threads(jobs=4):
        assert os.environ["OMP_NUM_THREADS"] == str(max(1, (os.cpu_count() or 1) // 4))
    assert "OMP_NUM_THREADS" not in os.environ

    monkeypatch.setenv("OMP_NUM_THREADS", "3")           # an explicit user choice is respected
    with limited_worker_threads(jobs=4):
        assert os.environ["OMP_NUM_THREADS"] == "3"


# --- Interruption through the real command line -------------------------------------------------------

INTERRUPT_CONFIG = {
    "sweep_name": "interrupt_test",
    "systems": [{"molecule": "h_chain", "n_atoms": 4, "geometries": {"start": 0.8, "stop": 2.4, "num": 5}}],
    "analyses": [{"name": "exact", "modes": ["exact"], "poly_degrees": [4, 8]},
                 {"name": "approx", "modes": ["approx"], "poly_degrees": [4, 8], "bound_extension": 0.05}],
}


def run_cli(arguments, **popen_options):
    return subprocess.Popen([sys.executable, "-m", "overlapanalyzer.sweep", *arguments], cwd=REPOSITORY_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL), **popen_options)


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_interrupted_sweep_resumes_to_identical_results(tmp_path, jobs):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(INTERRUPT_CONFIG))
    reference_dir, interrupted_dir = str(tmp_path / "reference"), str(tmp_path / "interrupted")

    reference = run_cli(["run", str(config_path), "-o", reference_dir])
    reference.communicate()
    assert reference.returncode == 0

    interrupted = run_cli(["run", str(config_path), "-o", interrupted_dir, "--jobs", jobs])
    time.sleep(3.0)
    interrupted.send_signal(signal.SIGINT)
    output, _ = interrupted.communicate(timeout=120)
    assert interrupted.returncode in (130, 0), output        # 0 only if the sweep finished before the signal
    assert not os.path.exists(os.path.join(interrupted_dir, ".sweep.lock"))

    resumed = run_cli(["run", str(config_path), "-o", interrupted_dir])
    output, _ = resumed.communicate(timeout=300)
    assert resumed.returncode == 0, output
    assert "failed" not in output.split("Done:")[-1].split(",")[0]
    expected, actual = bounds_table(reference_dir), bounds_table(interrupted_dir)
    assert len(expected) == len(actual) == 20
    assert_same_bounds(expected, actual)


# --- Command-line exit codes ---------------------------------------------------------------------------

def test_cli_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(TWO_ANALYSES))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"systems": [{"molecule": "h2", "geometries": [-1.0]}], "analyses": [{"name": "a"}]}))
    failing = tmp_path / "failing.json"
    failing.write_text(json.dumps({"systems": [{"molecule": "h2", "geometries": [0.74]}],
                                   "analyses": [{"name": "a", "target": {"strategy": "index", "index": 50}}]}))
    output_dir = str(tmp_path / "out")

    assert cli.main(["validate", str(bad)]) == 2
    assert cli.main(["run", str(good), "-o", output_dir]) == 0
    assert cli.main(["status", str(good), "-o", output_dir]) == 0
    assert cli.main(["run", str(failing), "-o", str(tmp_path / "out2")]) == 1
    with OutputDirectoryLock(output_dir):
        assert cli.main(["run", str(good), "-o", output_dir]) == 3


# --- Optional backends fail cleanly ------------------------------------------------------------------------

def test_mosek_without_license_fails_only_that_mode(run_config):
    pytest.importorskip("mosek")
    config, output_dir, _ = run_config({
        "systems": [{"molecule": "h2", "geometries": [0.74]}],
        "analyses": [{"name": "mixed", "modes": ["exact", "approx"], "poly_degrees": [2, 4],
                      "bound_extension": 0.05, "solver": "MOSEK"}],
    })
    result = load_result(output_dir, config.systems[0], config.analyses[0])
    assert result["bounds"]["exact"]["status"] == "complete"
    if result["bounds"]["approx"]["status"] == "failed":       # no MOSEK license on this machine
        assert result["status"] == "failed" and "license" in result["error"].lower()
    else:                                                       # licensed: MOSEK results must be valid too
        assert all(row["brackets_exact"] for row in result["bounds"]["approx"]["degrees"])


def test_ham_gen_backend_without_qcc_fails_cleanly(run_config):
    try:
        import qcc_240202  # noqa: F401
        pytest.skip("qcc_240202 is installed; this test covers its absence")
    except ImportError:
        pass
    config, output_dir, outcomes = run_config({
        "systems": [{"molecule": "h2", "geometries": [0.74], "backend": "ham_gen"}],
        "analyses": [{"name": "ground"}],
    })
    assert outcomes[0].state == "failed" and "qcc_240202" in outcomes[0].message
