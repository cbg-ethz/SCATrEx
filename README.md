<div align="left">
  <img src="figures/logo_text.png", width="200px">
</div>

[![pypi](https://img.shields.io/pypi/v/scatrex.svg)](https://pypi.python.org/pypi/scatrex)

Map single-cell transcriptomes to copy number evolutionary trees. Check out the [tutorial](./notebooks/tutorial.ipynb) for more information.

## Installation
```
$ pip install scatrex
```

SCATrEx uses [JAX](https://github.com/google/jax) to perform automatic differentiation. By default, SCATrEx installs the CPU-only version of JAX, but we strongly recommend the use of GPU acceleration. Please follow the instructions in https://github.com/google/jax#pip-installation-gpu-cuda to install the GPU version of JAX.
