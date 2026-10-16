"""
Command-line interface for molecular parameter sweeps.

    python -m overlapanalyzer.sweep validate CONFIG          # check the config and list the experiments
    python -m overlapanalyzer.sweep run CONFIG [options]     # run (or resume) the sweep
    python -m overlapanalyzer.sweep status CONFIG            # show what is complete, failed or pending
    python -m overlapanalyzer.sweep summarize CONFIG         # rebuild results.csv / bounds.csv / results.jsonl

The same commands are available as `overlap-sweep ...` after `pip install -e .`.
Exit codes: 0 all selected experiments complete, 1 some failed or pending, 2 invalid configuration, 130 interrupted.
"""
from __future__ import annotations

import argparse
import os
import sys

from overlapanalyzer.sweep.config import ConfigError, load_sweep_config
from overlapanalyzer.sweep.runner import plan_sweep, read_experiment_state, run_sweep, write_summaries


def resolve_output_dir(config, cli_output_dir: str | None) -> str:
    if cli_output_dir:
        return cli_output_dir
    if config.output_dir:
        return config.output_dir
    return os.path.join("sweeps", config.sweep_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="overlap-sweep", description="Automated molecular parameter sweeps for OverlapAnalyzer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser):
        subparser.add_argument("config", help="Path to the sweep configuration (JSON).")
        subparser.add_argument("-o", "--output", help="Output directory (default: config 'output_dir', else sweeps/<sweep_name>).")

    validate_parser = subparsers.add_parser("validate", help="Validate a config and list its experiments without running anything.")
    add_common_arguments(validate_parser)

    run_parser = subparsers.add_parser("run", help="Run or resume a sweep.")
    add_common_arguments(run_parser)
    run_parser.add_argument("-j", "--jobs", type=int, default=1, help="Systems to process in parallel (default 1).")
    run_parser.add_argument("--retry-failed", action="store_true", help="Recompute experiments that previously failed.")
    run_parser.add_argument("--force", action="store_true", help="Recompute everything selected, ignoring cached results.")
    run_parser.add_argument("--only", help="Only experiments whose system id or analysis name contains this text.")

    status_parser = subparsers.add_parser("status", help="Show the state of every experiment.")
    add_common_arguments(status_parser)

    summarize_parser = subparsers.add_parser("summarize", help="Rebuild the summary files from stored results.")
    add_common_arguments(summarize_parser)
    return parser


def print_experiment_list(config, output_dir: str) -> dict:
    counts = {}
    for system in config.systems:
        for analysis in config.analyses:
            state = read_experiment_state(output_dir, system, analysis)
            counts[state] = counts.get(state, 0) + 1
            print(f"  {state:<9} {system.system_id} :: {analysis.name}")
    return counts


def command_validate(config, output_dir: str) -> int:
    print(f"Sweep '{config.sweep_name}': {len(config.systems)} system(s) x {len(config.analyses)} analysis(es) "
          f"= {config.experiment_count} experiment(s); output -> {output_dir}")
    print_experiment_list(config, output_dir)
    return 0


def command_status(config, output_dir: str) -> int:
    counts = print_experiment_list(config, output_dir)
    print("Totals: " + ", ".join(f"{state} {count}" for state, count in sorted(counts.items())))
    return 0 if set(counts) == {"complete"} else 1


def command_run(config, output_dir: str, arguments) -> int:
    work_items = plan_sweep(config, output_dir, arguments.retry_failed, arguments.force, arguments.only)
    total = sum(len(work.analyses) + len(work.skipped) for work in work_items)
    to_compute = sum(len(work.analyses) for work in work_items)
    print(f"Sweep '{config.sweep_name}' -> {output_dir}: {total} experiment(s) selected, {to_compute} to compute, "
          f"{total - to_compute} cached.")
    progress = {"done": 0}

    def report(outcome):
        progress["done"] += 1
        timing = f" ({outcome.seconds:.1f} s)" if outcome.seconds else ""
        detail = f"  -> {outcome.message}" if outcome.state in ("failed", "cached-failed") and outcome.message else ""
        print(f"[{progress['done']:>{len(str(total))}}/{total}] {outcome.state:<15} {outcome.label}{timing}{detail}", flush=True)

    try:
        outcomes = run_sweep(config, output_dir, jobs=arguments.jobs, retry_failed=arguments.retry_failed,
                             force=arguments.force, only=arguments.only, report=report)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed experiments are saved; run the same command again to resume.")
        return 130

    failed = [outcome for outcome in outcomes if outcome.state in ("failed", "cached-failed")]
    print(f"Done: {len(outcomes) - len(failed)} complete, {len(failed)} failed. "
          f"Summaries: {os.path.join(output_dir, 'results.csv')}, {os.path.join(output_dir, 'bounds.csv')}")
    if any(outcome.state == "cached-failed" for outcome in failed):
        print("Previously failed experiments were skipped; use --retry-failed to recompute them.")
    return 1 if failed else 0


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_sweep_config(arguments.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    output_dir = resolve_output_dir(config, arguments.output)

    if arguments.command == "validate":
        return command_validate(config, output_dir)
    if arguments.command == "status":
        return command_status(config, output_dir)
    if arguments.command == "summarize":
        counts = write_summaries(config, output_dir)
        print(f"Summaries rebuilt in {output_dir}: " + ", ".join(f"{state} {count}" for state, count in sorted(counts.items())))
        return 0
    return command_run(config, output_dir, arguments)


if __name__ == "__main__":
    sys.exit(main())
