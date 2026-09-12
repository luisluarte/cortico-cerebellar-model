---
name: stan-cognitive-modeling
description: >-
  Authoring, optimizing, and evaluating Bayesian cognitive models in Stan (CmdStanR / CmdStanPy / PyStan).
  Covers the wiener_lpdf likelihood, hierarchical parameter structures, variational inference (Pathfinder),
  L-BFGS mode estimation, NUTS HMC sampling diagnostics, and Leave-One-Subject-Out (LNSO) cross-validation.
---

# Bayesian Cognitive Modeling in Stan

## 1. The Wiener Likelihood in Stan

Stan provides native support for the 4-parameter Wiener diffusion model:
```stan
target += wiener_lpdf(rt | a, tau, beta, delta);
```
Where:
* `rt`: Reaction time in seconds ($rt > \tau$).
* `a`: Boundary separation ($a > 0$).
* `tau`: Non-decision time ($0 \le \tau < \min(rt)$).
* `beta`: Starting point bias relative to $a$ ($\beta \in (0, 1)$).
* `delta`: Drift rate ($\delta \in \mathbb{R}$).

### 1.1 Choice Coding Convention
By mathematical convention, reaction times for choice $A = 1$ (upper boundary) are entered as positive values, while choice $A = 0$ (lower boundary) can be handled by inverting parameters:
$$\text{Lower boundary likelihood} = \text{wiener\_lpdf}(rt \mid a, \tau, 1 - \beta, -\delta)$$

---

## 2. Model Structure & Performance Best Practices

```stan
data {
  int<lower=1> N;                         // Number of trials
  vector<lower=0>[N] rt;                  // Observed RTs
  int<lower=0, upper=1> choice[N];        // Binary decisions
  int<lower=1> K;                         // Feature dimension
  matrix[N, K] X;                         // Covariates (e.g. reservoir states)
}

parameters {
  vector[K] w_drift;                      // Drift mapping weights
  real b_drift;                           // Drift intercept
  real<lower=0> a;                        // Boundary separation
  real<lower=0, upper=min(rt)> tau;       // Non-decision time
  real<lower=0, upper=1> beta;            // Bias
}

transformed parameters {
  vector[N] delta = X * w_drift + b_drift;
}

model {
  // Priors
  w_drift ~ normal(0, 1);
  b_drift ~ normal(0, 2);
  a ~ gamma(3, 3);
  tau ~ uniform(0, min(rt));
  beta ~ beta(2, 2);

  // Vectorized / trial-level likelihood
  for (i in 1:N) {
    if (choice[i] == 1) {
      target += wiener_lpdf(rt[i] | a, tau, beta, delta[i]);
    } else {
      target += wiener_lpdf(rt[i] | a, tau, 1.0 - beta, -delta[i]);
    }
  }
}
```

---

## 3. Inference Engines & Diagnostics

### 3.1 Inference Engine Selection
* **L-BFGS Optimization**: Rapid point estimation for screening hundreds of model topologies in seconds.
* **Pathfinder**: Variational inference using quasi-Newton optimization to fit Gaussian approximations on challenging multi-modal posteriors. Substantially faster than full MCMC for complex high-dimensional reservoir readouts.
* **NUTS (No-U-Turn Sampler)**: Full Bayesian posterior exploration. Standard settings: 4 chains, 1000 warmup, 1000 sampling iterations.

### 3.2 Convergence Criteria
* **R-hat ($\hat{R}$)**: Must be $< 1.01$ across all parameters (Gelman-Rubin split-$\hat{R}$).
* **Effective Sample Size (ESS)**: Both Bulk-ESS and Tail-ESS must exceed 400.
* **Divergences**: Zero divergences tolerated. If divergences occur:
  1. Increase `adapt_delta` (e.g. from 0.8 to 0.95 or 0.99).
  2. Non-center hierarchical priors: $\theta_{\text{subj}} = \mu + \sigma \cdot \theta_{\text{raw}}$, where $\theta_{\text{raw}} \sim \mathcal{N}(0, 1)$.
* **E-BFMI**: Energy Bayesian Fraction of Missing Information must exceed 0.3.

---

## 4. Model Evaluation & Out-of-Sample Validation

### 4.1 Leave-One-Subject-Out (LNSO)
Partition subject cohort into $K$ folds. Fit on $K-1$ subjects, evaluate log-predictive density and CRPS on held-out subject:
$$\text{LNSO-LL} = \sum_{s=1}^S \sum_{i \in \text{Trials}_s} \log p(\text{RT}_i, A_i \mid \hat{\theta}_{-s})$$

### 4.2 Posterior Predictive Checks (PPC)
Generate simulated RT and choice distributions from posterior draws:
* Compute RT quantiles (10%, 30%, 50%, 70%, 90%) for both upper and lower boundaries.
* Compare observed empirical quantiles against 95% posterior credible intervals.
