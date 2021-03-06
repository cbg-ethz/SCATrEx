<div align="left">
  <img src="https://github.com/cbg-ethz/SCATrEx/raw/main/figures/scatrex.png", width="300px">
</div>
<p></p>

[![PyPI](https://img.shields.io/pypi/v/scatrex.svg?style=flat)](https://pypi.python.org/pypi/scatrex)
[![Build](https://github.com/cbg-ethz/SCATrEx/actions/workflows/main.yaml/badge.svg)](https://github.com/cbg-ethz/SCATrEx/actions/workflows/main.yaml)


Map single-cell transcriptomes to copy number evolutionary trees. Check out the [tutorial](https://github.com/cbg-ethz/SCATrEx/blob/main/notebooks/tutorial.ipynb) for more information.

## Installation
```
$ pip install scatrex
```

SCATrEx uses [JAX](https://github.com/google/jax) to perform automatic differentiation. By default, SCATrEx installs the CPU-only version of JAX, but we strongly recommend the use of GPU acceleration. Please follow the instructions in https://github.com/google/jax#pip-installation-gpu-cuda to install the GPU version of JAX.

## Preprint
https://doi.org/10.1101/2021.11.04.467244
