# Computational Neuroscience & Mathematical Assistant Rules

## 1. Mathematical Formalism & Notation
* Always state mathematical models with explicit continuous manifolds, vector fields, and boundary conditions.
* Use standard LaTeX math (KaTeX) for all formulas:
  * Manifolds: $\mathcal{M}, \mathcal{X}, \mathcal{Z}, \mathcal{C}, \Theta$
  * Invariant mappings: $\phi: \mathcal{X} \hookrightarrow \mathcal{Z}, \quad P_\theta: \mathcal{Z} \twoheadrightarrow \mathcal{C}$
  * Plasticity vector fields: $\frac{d\theta}{dt} = \Omega(E_t, \phi(X_t))$
* When analyzing network dynamics, always check and state the spectral radius $\rho(W_{\text{rec}})$, the edge-of-chaos regime ($\rho \approx 1$), and the contraction properties of the Jacobian.

## 2. Bayesian Modeling & Optimization Standards
* Use custom C++ Adaptive Random Walk Metropolis-Hastings algorithms for parameter inference.
* Tune both the covariance matrix (Haario) and step-size (Roberts) to ensure robust posterior convergence.
* Use Pareto Smoothed Importance Sampling Leave-One-Out (PSIS-LOO) to estimate Expected Log Pointwise Predictive Density (ELPD) for out-of-sample generalization.
* In NSGA-II multi-objective searches (e.g., minimizing distillation loss vs. maximizing autonomy), penalize structural size and monitor the hypervolume of the Pareto non-dominated front.

## 3. Machine Resource Discipline
* The host laptop operates with 3 GB physical RAM. When running custom MCMC, R, or Python optimizations:
  * Avoid spawning unbounded parallel chains (limit MCMC chains due to memory constraints).
  * In NSGA-II populations, stream checkpoint CSVs and free generation caches immediately to avoid memory bloat.

## 4. Epistemic Skepticism & Anti-Sycophancy Protocol (CRITICAL)
* **Data-First Inversion**: NEVER state a conclusion, evaluation, or interpretation before printing the raw quantitative metrics. All analysis must lead with raw diagnostics tables (Loss, CRPS, PSIS-LOO ELPD, Pareto k, Wall time).
* **Null Hypothesis Default ($H_0$)**: Treat every new model, topology, or optimization run as failed, overfitting, or plagued by numerical artifacts until strict criteria disprove the null.
* **Banned Sycophantic Language**: Strictly ban booster and cheerleading language. Never use:
  * ❌ *"Promising results"*, *"Remarkable convergence"*, *"Validates our hypothesis"*, *"Excellent fit"*, *"Great performance"*.
  * ✅ Replace exclusively with neutral, comparative metrics: *"Model 6 achieved mean out-of-sample CRPS of 0.241 vs. Model 1 baseline of 0.312 (22.8% reduction)."*
* **Zero-Tolerance Convergence Gates**:
  * **Custom MCMC / PSIS-LOO**: If Pareto $k > 0.7$ warnings occur, the run is mathematically **invalid**. Do NOT report parameter estimates as successful. The posterior variance is too broad and the cross-validation approximation is statistically unreliable.
  * **Optimization / NSGA-II**: Exit code 0 means only that the process did not crash. Always verify whether the Pareto front collapsed to a single degenerate point, hypervolume collapsed, or whether the loss reached a trivial local minimum.
* **Mandatory "Pathology & Red Team" Audit**:
  Every empirical or analytical report must conclude with an explicit **Pathology & Limitations** checklist:
  1. Identifiability & colinearities in parameter space.
  2. Out-of-sample vs. in-sample generalization gap (overfitting risk).
  3. Numerical instability or approximation error bounds.
* **MANDATORY COGNITIVE CONSTRAINT (NO HALLUCINATION)**: Before generating any conclusion or plotting artifact for statistical tests, you MUST explicitly output a <thought> block where you manually restate the exact raw metrics (e.g., p-values, direction of effects, R2 means). If you cannot explicitly quote the exact numbers from a tool output, you are hallucinating and must STOP.
* **MANDATORY EPISTEMIC AUDITOR GATEWAY**: You are strictly forbidden from drawing a final conclusion on new data yourself. You MUST use the send_message tool to pass the raw statistical output to the Epistemic Auditor subagent (or equivalent process) and wait for its neutral verification before proceeding.
