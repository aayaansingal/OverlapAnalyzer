"""
Automated parameter sweeps over molecular systems for OverlapAnalyzer.

A sweep expands a JSON configuration into (system, analysis) experiments:

- a *system* fixes the molecule, geometry, basis, reference spin and active space, and owns the
  expensive stages: Hamiltonian generation and exact diagonalization (eigen.calc_eigen);
- an *analysis* fixes how a target eigenstate is chosen and which polynomial overlap bounds
  (polynomial_majorisation_LB.OverlapEstimator) are computed from the Hamiltonian moments.

Systems are cached independently of analyses, so changing bound settings never repeats a
diagonalization. Run `python -m overlapanalyzer.sweep --help` for the command-line interface.
"""
from overlapanalyzer.sweep.config import load_sweep_config, SweepConfig, SystemSpec, AnalysisSpec, ConfigError
from overlapanalyzer.sweep.runner import run_sweep, plan_sweep
