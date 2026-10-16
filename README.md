Overlap analysis module
===============================

This repository continues development of [OverlapAnalyzer](https://github.com/JunanLin/OverlapAnalyzer), whose underlying scientific software was developed mainly by Junan Lin. Its history up to `1f1510c` (2025-12-08) is taken directly from that repository.

Library install
---------------

After downloading the repository, it can be installed as a Python library with:

```
  pip install -r requirements.txt
  pip install -e .
```
The dependency on QuantumMAMBO needs some extra care since the juliacall package will by default install the latest version of Julia in your environment, which will not be compatible with QuantumMAMBO. A temporary resolution is as follows (assuming the virtual environment is located at /.venv):

(1) Locate the .venv/julia_env/pyjuliapkg folder

(2) Copy the juliapkg.json file to that folder

(3) Run
```
python setup-julia.py
```

Some dependencies need separate handling:

- `qcc_240202` (iQCC/QCC routines used by `ham_gen`, `ham_conversion` and `iQCC`) is not on PyPI and must be installed separately.
- `Mosek` (used by `polynomial_majorisation_LB` through `cvxpy`) needs a MOSEK license file.
- `krypy` is only needed for `alg_LinearOp.calculateInnerProducts` and `precond_utils`. Its latest release (2.2.0) does not import on recent SciPy versions.

Usage
-----
Calculating wavefunction overlaps with Hamiltonian eigenstates from Hamiltonian power expectation values.
arXiv: https://arxiv.org/abs/2503.12224
