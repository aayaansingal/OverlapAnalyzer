import json

import pytest

from overlapanalyzer.sweep.config import parse_sweep_config
from overlapanalyzer.sweep.runner import run_sweep
from overlapanalyzer.sweep.storage import SystemPaths


def load_result(output_dir, system, analysis):
    with open(SystemPaths(output_dir, system.system_id).result_json(analysis.analysis_id)) as handle:
        return json.load(handle)


@pytest.fixture
def run_config(tmp_path):
    """Parse a config dict, run it in a temporary directory, and return (config, output_dir, outcomes)."""
    def runner(raw_config, output_dir=None, **run_options):
        config = parse_sweep_config(raw_config)
        directory = str(output_dir or tmp_path)
        outcomes = run_sweep(config, directory, **run_options)
        return config, directory, outcomes
    return runner
