from setuptools import setup, find_packages

# Core dependencies; optional features are installed through extras (see README)
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

short_description = \
    """
    Bounds on eigenstate overlaps from Hamiltonian moments, with automated molecular parameter sweeps.
    """

setup(
    name='overlapanalyzer',
    version='0.4.0',
    author='Junan Lin',
    author_email='junan_lin@hotmail.com',
    maintainer='Aayaan Jain Singal',
    description=short_description.strip(),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aayaansingal/OverlapAnalyzer',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=requirements,
    extras_require={
        'dev': ['pytest'],
        'julia': ['juliacall', 'juliapkg'],    # partitioning.py (QuantumMAMBO via Julia)
        'krylov': ['krypy'],                   # precond_utils.py; krypy 2.2.0 does not import on recent SciPy
        'iqcc': ['pathos'],                    # iQCC.py; also needs the non-public qcc_240202 package
    },
    entry_points={
        'console_scripts': ['overlap-sweep=overlapanalyzer.sweep.cli:main'],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.11",
)
