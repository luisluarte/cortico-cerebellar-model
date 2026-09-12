---
name: information-geometry-and-optimization
description: >-
  Mathematical foundations of Information Geometry, the Information Bottleneck principle,
  Continuous Ranked Probability Score (CRPS), and multi-objective Pareto optimization (NSGA-II).
  Covers Riemannian statistical manifolds, Fisher Information Metrics, Natural Gradients,
  and evolving neural architectures across functional manifolds.
---

# Information Geometry & Mathematical Optimization

## 1. Information Geometry on Statistical Manifolds

A statistical model $S = \{p(x \mid \theta) : \theta \in \Theta \subseteq \mathbb{R}^m\}$ forms a differential manifold equipped with a Riemannian metric tensor.

### 1.1 The Fisher Information Metric (FIM)
The Fisher Information defines the canonical Riemannian metric $g_{ij}(\theta)$ on the statistical manifold:
$$g_{ij}(\theta) = \mathbb{E}_{p(x|\theta)} \left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right] = -\mathbb{E}_{p(x|\theta)} \left[ \frac{\partial^2 \log p(x|\theta)}{\partial \theta_i \partial \theta_j} \right]$$

### 1.2 Invariant Distance (Rao Distance)
The infinitesimal distance between two nearby probability distributions separated by $d\theta$ is given by the Fisher-Rao line element:
$$ds^2 = \sum_{i,j} g_{ij}(\theta) d\theta_i d\theta_j = 2 D_{\text{KL}}(p_\theta \parallel p_{\theta + d\theta}) + \mathcal{O}(\|d\theta\|^3)$$
This metric is strictly invariant under any smooth, invertible reparameterization of both the sample space and the parameter manifold.

### 1.3 Natural Gradient Descent
Standard Euclidean gradients $\nabla_\theta L$ depend arbitrarily on parameter scaling. The Natural Gradient $\tilde{\nabla} L$ defines the steepest descent direction with respect to the intrinsic Riemannian manifold:
$$\tilde{\nabla}_\theta L = G^{-1}(\theta) \nabla_\theta L$$
$$\theta_{t+1} = \theta_t - \eta G^{-1}(\theta_t) \nabla_\theta L(\theta_t)$$

---

## 2. The Information Bottleneck in Neural Compression

When a high-dimensional expansion $\mathcal{Z} = \phi(\mathcal{X})$ is compressed into a readout representation $\mathcal{C} = P_\theta(\mathcal{Z})$ to predict an action context or behavioral distribution $\mathcal{Y}$:

$$\min_{P_\theta} \mathcal{L}_{\text{IB}} = I(\mathcal{Z}; \mathcal{C}) - \beta I(\mathcal{C}; \mathcal{Y})$$
where:
* $I(\mathcal{Z}; \mathcal{C}) = H(\mathcal{C}) - H(\mathcal{C} \mid \mathcal{Z})$ measures the complexity / rate of the compression.
* $I(\mathcal{C}; \mathcal{Y})$ is the preserved predictive information about the behavioral output.
* $\beta$ is the Lagrange multiplier governing the trade-off between representational parsimony and task fidelity.

---

## 3. Continuous Ranked Probability Score (CRPS)

The CRPS is a strictly proper scoring rule that evaluates full probabilistic forecast distributions $F$ against empirical observations $y \in \mathbb{R}$:
$$\text{CRPS}(F, y) = \int_{-\infty}^\infty \left( F(t) - \mathbb{I}(t \ge y) \right)^2 dt$$

### 3.1 Closed-Form Representations
$$\text{CRPS}(F, y) = \mathbb{E}_{X \sim F}[|X - y|] - \frac{1}{2} \mathbb{E}_{X, X' \sim F}[|X - X'|]$$
* Penalizes both distribution miscalibration (bias) and overconfidence / excessive dispersion.
* Crucial for reaction time distributions where tail behavior (outliers, non-decision variance) skews standard mean squared error.

---

## 4. Multi-Objective Evolutionary Optimization (NSGA-II)

When searching a functional manifold of architectures $\mathcal{M}$ where objectives conflict (e.g. maximizing Evidence $\log p(\text{RT}, A \mid m)$ while minimizing Complexity / CRPS error):

### 4.1 Pareto Dominance
Solution $m_1$ dominates $m_2$ ($m_1 \succ m_2$) if:
$$\forall i \in \{1, \dots, K\}, \quad f_i(m_1) \le f_i(m_2) \quad \text{and} \quad \exists j, \quad f_j(m_1) < f_j(m_2)$$

### 4.2 Non-Dominated Sorting & Crowding Distance
1. Partition population $\mathcal{P}$ into non-dominated Pareto fronts $\mathcal{F}_1, \mathcal{F}_2, \dots$
2. For individuals on the same front, compute crowding distance $I_{\text{distance}}$ along each objective axis:
   $$I_{\text{distance}}(i) = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$
3. Selection favors lower non-domination rank $\mathcal{F}_k$, and breaks ties using higher crowding distance to maintain diverse topological exploration.
