# Optimized Gauss-Hermite Quadrature

Compute $\mathbb{E}[f(X)]$ for $X \sim \mathcal{N}(\mu, \Sigma)$ using optimized Gauss-Hermite quadrature with adaptive parameter optimization.

## Installation

```bash
pip install optimized-gauss-hermite
```

## Quick start

```python
import numpy as np
from ogh_integration import ogh_integrate

Ex = np.array([0.5, -0.1])
Cx = np.array([[2.0, 0.5],
               [0.5, 1.0]])

def f(x):
    return np.prod(np.cos(x)**2, axis=0)

result = ogh_integrate(f, Ex, Cx)
print(f"E[f(X)] = {result:.12f}")
```

## Packages

| Package | Purpose |
|---|---|
| `gauss_hermite_quadrature` | Nodes, weights, full-grid and sparse-grid Gauss-Hermite rules |
| `optimal_gauss_hermite` | Optimized quadrature with adaptive parameter optimization |
| `ogh_integration` | High-level `ogh_integrate()` convenience wrapper |

## API

```python
ogh_integrate(
    func,                     # f(X) where X is (dimX, N) → (N,)
    Ex,                       # mean (dimX,)
    Cx,                       # covariance (dimX, dimX), SPD
    *,                        # all remaining args must be named
    sample_num=None,          # sample points for optimization (auto if None)
    num_quad=3,               # quadrature level
    sparse_mode=False,        # True for sparse-grid (high-dim problems)
    poly_type="hermite",      # polynomial basis
    sample_type="latin",      # sampling: "latin", "sobol", "haltonset", "MC"
    algorithm_type="levenberg-marquardt",  # "hybrid" or "hessian"
    decomp_type="Chol",       # "Chol" or "LDL"
)
```

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20
- scipy ≥ 1.7

## Citation

```bibtex
@inproceedings{ogh2024,
  title     = {Optimized Gauss-Hermite Quadrature: A Refined Approach},
  booktitle = {2024 IEEE Conference on Decision and Control (CDC)},
  author    = {Meng, Haozhan},
  year      = {2024},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/abstract/document/10590620}
}
```

## License

MIT
