# Quantum Repeater simulation with multiple satellites

[![DOI](https://zenodo.org/badge/421734256.svg)](https://zenodo.org/badge/latestdoi/421734256)

This repository is an archive for the code used in:

> Simulating quantum repeater strategies for multiple satellites <br>
> J. Wallnöfer, F. Hahn, M. Gündoğan, J. S. Sidhu, F. Krüger, N. Walk, J. Eisert, J. Wolters <br>
> Preprint: [arXiv:2110.15806 [quant-ph]](https://arxiv.org/abs/2110.15806)

## Repository structure

This repository contains two main parts (that are however not completely separated yet in this version of the simulation code):

* The core of the simulation consisting of the .py files in the repository root and the `libs` directory.
* Multiple simulation setups with files that set up, run and evaluate the scenarios in the `scenarios` directory

### How to use

If you wish to run the scenarios yourself, we recommend recreating the same virtual environment we used to develop the code via pipenv. 
This assumes a version of Python 3.8 is available on your system. 

```bash
pip install pipenv
pipenv sync
```

The scenario files are expected to be called from the main directory, e.g. to run an example configuration that outputs runtime and event stats use the following in the main directory: 
```bash
pipenv run python scenarios/three_satellites/twolink_downlink.py
```

The scenario files that begin with `run_` are meant to be run on a cluster computer with multiple processors. If that is not your usecase please adjust them accordingly.

If you want to use the plot files use `pipenv sync --dev` instead (separate because some graphic libraries may not be available on HPC systems).


## Related projects

An updated version of the core part of the simulation code has since been released separately and is available as a Python package: [https://github.com/jwallnoefer/requsim](https://github.com/jwallnoefer/requsim) 
