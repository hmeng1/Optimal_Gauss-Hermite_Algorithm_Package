"""Basic usage examples for ogh_integrate."""

import numpy as np
from ogh_integration import ogh_integrate

dimX = 3
np.random.seed(42)
Ex = np.random.randn(dimX)
Csqrt = np.random.randn(dimX, dimX)
Cx = Csqrt @ Csqrt.T

print(f"dimX = {dimX}")
print(f"Ex   = {Ex}")
print(f"Cx   =\n{Cx}")
print()

# Example 1: E[prod_i cos^2(X_i)]
def f1(x):
    return np.prod(np.cos(x) ** 2, axis=0)

v1 = ogh_integrate(f1, Ex, Cx, num_quad=3)
print(f"E[prod cos^2(X_i)]   = {v1:.12f}")

# Example 2: E[cos(sum_i X_i)]
def f2(x):
    return np.cos(x.sum(axis=0))

v2 = ogh_integrate(f2, Ex, Cx, num_quad=3)
print(f"E[cos(sum X_i)]      = {v2:.12f}")

# Example 3: E[exp(-||X||^2 / 4)]
def f3(x):
    return np.exp(-0.25 * np.sum(x ** 2, axis=0))

v3 = ogh_integrate(f3, Ex, Cx, num_quad=3)
print(f"E[exp(-||X||^2/4)]   = {v3:.12f}")
