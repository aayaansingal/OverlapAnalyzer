from setuptools import setup, find_packages

# Parse requirements from file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

short_description = \
    """
    A library for evaluating initial state overlap for quantum algorithms.
    """

setup(
    name='overlapanalyzer',
    version='0.1.0',
    author='Junan Lin',
    author_email='junan_lin@hotmail.com',
    description=short_description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': ['overlap-sweep=overlapanalyzer.sweep.cli:main'],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)