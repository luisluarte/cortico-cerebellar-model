---
name: sympy
description: Exact symbolic mathematics, Jacobian linearizations, eigenvalues, bifurcations, and LaTeX generation with SymPy.
---

# SymPy for Mathematical Foundations

## 1. Symbolic Variables & System Jacobians
```python
import sympy as sp

x, y, v_in = sp.symbols("x y v_in", real=True)
tau, w_rec, w_in = sp.symbols("tau w_rec w_in", positive=True)

# 2D Dynamical System
f1 = (-x + sp.tanh(w_rec * x - y + w_in * v_in)) / tau
f2 = (-y + sp.tanh(x + y)) / tau

F = sp.Matrix([f1, f2])
state_vars = sp.Matrix([x, y])

# Analytical Jacobian Matrix
J = F.jacobian(state_vars)
```

## 2. Eigenvalues & Fixed Point Stability
```python
# Fixed points when F = 0
fixed_points = sp.solve([f1, f2], (x, y))

# Evaluate Jacobian at origin (0, 0)
J_origin = J.subs([(x, 0), (y, 0), (v_in, 0)])
eigenvalues = J_origin.eigenvals()
```

## 3. Fast Numerical Evaluation (Lambdify)
```python
# Convert symbolic expression to optimized NumPy / SciPy callable
f_numeric = sp.lambdify((x, y, v_in, tau, w_rec, w_in), J, modules="numpy")
```

## 4. LaTeX Generation for Reports
```python
latex_str = sp.latex(J)
```
