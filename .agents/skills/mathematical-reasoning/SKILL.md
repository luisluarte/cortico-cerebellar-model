---
name: mathematical-reasoning
description: Rigorous analytical mathematical reasoning, formal derivations, asymptotic perturbation methods, dynamical systems stability, stochastic calculus, and automated symbolic proof verification.
---

# Mathematical Reasoning & Analytical Derivation Engine

A systematic protocol for conducting formal mathematical derivations, stability proofs, asymptotic expansions, and stochastic analysis with zero hand-waving and mandatory algebraic verification.

---

## 1. Universal Derivation Protocol

Whenever deriving mathematical formulations, proving stability, or solving differential equations, execute the five-stage pipeline:

```
[1. Problem Formalization] ──> [2. Explicit Assumptions] ──> [3. Step-by-Step Derivation]
                                                                        │
                                                                        ▼
[5. Symbolic Verification] <── [4. Asymptotic & Boundary Checks] <──────┘
```

### Stage 1: Formal Problem Statement
* State the state space $\mathcal{M} \subseteq \mathbb{R}^n$ and parameter manifold $\Theta$.
* Declare the exact governing system: ODE $\dot{x} = f(x, \theta, t)$, SDE $\mathrm{d}x_t = \mu(x_t)\mathrm{d}t + \sigma(x_t)\mathrm{d}W_t$, or PDE $\partial_t p = \mathcal{L} p$.
* Define boundary conditions $\mathcal{B}[x] = 0$ and initial data $x(0) = x_0$.

### Stage 2: Explicit Assumption Register
Explicitly document regularity conditions before proceeding:
1. **Regularity**: Lipschitz continuity $L$, smoothness class $C^k$, or compactness of invariant sets $K \subset \mathcal{M}$.
2. **Spectral properties**: Definiteness of matrices ($A + A^T \prec 0$), singular values $\sigma_{\min}, \sigma_{\max}$, or spectral radius $\rho(W) < 1$.
3. **Stochastic regime**: Martingale conditions, Novikov condition for Girsanov measure transforms, or boundary classifications (absorbing, reflecting, natural).

### Stage 3: Step-by-Step Derivation
* **No skipped steps**: Show intermediate tensor contractions, integration by parts, and boundary terms explicitly.
* **Notation hygiene**: Distinguish Itô integrals $\int \circ \mathrm{d}W$ vs Stratonovich $\int * \mathrm{d}W$; distinguish coordinate charts from intrinsic geometric quantities.
* **Conservation laws**: Explicitly check conservation of probability ($\int_\Omega p(x,t)\mathrm{d}x = 1 - P_{\text{absorbed}}(t)$) or energy.

### Stage 4: Asymptotic & Boundary Limit Checks
Validate every analytical result against extreme limiting cases:
* **Short-time limit**: $t \to 0^+$ (singular Gaussian diffusion dominance).
* **Long-time limit**: $t \to \infty$ (stationary distribution $\pi(x)$ or absorption limit).
* **Parameter boundaries**: $\sigma \to 0$ (deterministic dynamical limit), $\mu \to 0$ (pure unbiased Brownian motion), or $\theta \to \theta_c$ (bifurcation threshold).
* **Dimensional invariance**: Ensure all derived exponents and terms are strictly dimensionless.

### Stage 5: Mandatory Symbolic Verification
Every derived closed-form solution or matrix identity **must** be verified using the accompanying automated verification script (`scripts/verify_derivation.py` or inline `sympy`).

---

## 2. Dynamical Systems & Nonlinear Stability Analysis

### Contraction Analysis (Finsler-Lyapunov Metrics)
For non-autonomous neural dynamics $\dot{x} = f(x, t)$:
1. Compute the generalized Jacobian $J(x, t) = \frac{\partial f}{\partial x}$.
2. For a Riemannian metric tensor $M(x, t) = \Theta(x, t)^T \Theta(x, t) \succ 0$, define the generalized contraction metric:
   $$F = \left( \dot{\Theta} + \Theta J \right) \Theta^{-1}$$
3. Compute the generalized matrix measure (symmetric part):
   $$M_{\text{sym}} = \frac{1}{2}\left( J^T M + M J + \dot{M} \right)$$
4. If $\exists c > 0$ such that $M_{\text{sym}} \preceq -c M$ uniformly for all $x \in \mathcal{M}$, all trajectories exponentially converge to a unique trajectory at rate $c$:
   $$\|\delta x(t)\|_M \le \|\delta x(0)\|_M e^{-c t}$$

### Center Manifold Reduction
Near non-hyperbolic equilibria $x^*$ with Jacobian eigenvalues $\text{Re}(\lambda_c) = 0$ (center) and $\text{Re}(\lambda_s) < 0$ (stable):
1. Decouple coordinates into center $u \in \mathbb{R}^{n_c}$ and stable $v \in \mathbb{R}^{n_s}$:
   $$\dot{u} = A_c u + g_1(u, v), \quad \dot{v} = A_s v + g_2(u, v)$$
2. Express the center manifold as an invariant graph $v = h(u)$ with $h(0) = 0, Dh(0) = 0$.
3. Solve the invariance PDE:
   $$\mathcal{N}[h(u)] = Dh(u)[A_c u + g_1(u, h(u))] - A_s h(u) - g_2(u, h(u)) = 0$$
4. Approximate $h(u)$ by multivariate polynomial series expansion to determine reduced normal form dynamics.

---

## 3. Stochastic Calculus & First-Passage Density Derivations

### Kolmogorov Backward Equation for First-Passage
For an Itô diffusion $\mathrm{d}x_t = \mu(x)\mathrm{d}t + \sigma(x)\mathrm{d}W_t$ with absorbing thresholds at $\{0, a\}$:
1. The generator $\mathcal{L}^*$ acting on initial state $x_0 = x \in (0, a)$:
   $$\mathcal{L}^* u(x) = \mu(x) \frac{\partial u}{\partial x} + \frac{1}{2}\sigma(x)^2 \frac{\partial^2 u}{\partial x^2}$$
2. The survival probability $S(t | x) = \mathbb{P}(\tau > t | x_0 = x)$ satisfies:
   $$\frac{\partial S}{\partial t} = \mathcal{L}^* S, \quad S(0 | x) = 1, \quad S(t | 0) = S(t | a) = 0$$
3. The first-passage density is $g(t | x) = -\frac{\partial S(t | x)}{\partial t}$.
4. In the Laplace domain $\hat{g}(s | x) = \mathbb{E}[e^{-s \tau} | x]$, convert the PDE into an ordinary boundary value problem:
   $$s \hat{g}(s | x) - \mathcal{L}^* \hat{g}(s | x) = 0, \quad \hat{g}(s | a) = 1, \quad \hat{g}(s | 0) = 0$$

### Navarro-Fuss Series Switching Proof
For standard Wiener diffusion with drift $v$, boundary $a$, and diffusion coefficient $\sigma = 1$:
* **Large-time expansion** (Fourier spectral series, fast convergence for $t > 2a^2/\pi^2$):
  $$f_{\text{large}}(t) = \frac{\pi}{a^2} \exp\left( v a w - \frac{v^2 t}{2} \right) \sum_{k=1}^\infty k \sin(k \pi w) \exp\left( -\frac{k^2 \pi^2 t}{2 a^2} \right)$$
* **Short-time expansion** (Method of images, fast convergence for $t < 2a^2/\pi^2$):
  $$f_{\text{short}}(t) = \frac{1}{\sqrt{2\pi t^3}} \exp\left( v a w - \frac{v^2 t}{2} \right) \sum_{j=-\infty}^\infty (2j a + a w) \exp\left( -\frac{(2j a + a w)^2}{2 t} \right)$$
* Switching bound: Choose series to truncate at $k \le K(\epsilon)$ where truncation error $\le \epsilon$.

---

## 4. Asymptotic Expansions & Perturbation Theory

### Matched Asymptotic Expansions (Boundary Layers)
For singularly perturbed systems $\epsilon \ddot{y} + a(x)\dot{y} + b(x)y = 0$ with $0 < \epsilon \ll 1$:
1. **Outer solution** ($y_{\text{outer}}$): Expand $y = y_0(x) + \epsilon y_1(x) + O(\epsilon^2)$. Solves the reduced equation $a(x)\dot{y}_0 + b(x)y_0 = 0$.
2. **Inner solution** ($y_{\text{inner}}$): Introduce stretched boundary-layer coordinate $\xi = (x - x_b)/\epsilon^\alpha$. Balance dominant terms to determine $\alpha$.
3. **Prandtl Matching Principle**:
   $$\lim_{x \to x_b} y_{\text{outer}}(x) = \lim_{\xi \to \infty} y_{\text{inner}}(\xi)$$
4. **Composite Expansion**: Form the uniformly valid approximation:
   $$y_{\text{composite}}(x) = y_{\text{outer}}(x) + y_{\text{inner}}\left(\frac{x - x_b}{\epsilon^\alpha}\right) - y_{\text{overlap}}$$

---

## 5. Verification Checklist

Before accepting any mathematical derivation:
- [ ] Are all indices matched in tensor / matrix operations?
- [ ] Is the Jacobian evaluated at the correct stationary point $x^*$ (not general $x$)?
- [ ] Did you check whether noise is additive or multiplicative before applying Itô lemma?
- [ ] Are all probability density functions normalized ($\int_{-\infty}^\infty f(t)\mathrm{d}t = 1$)?
- [ ] Does the short-time expansion match the method-of-images Gaussian kernel as $t \to 0$?
- [ ] Has the result been verified symbolically via `verify_derivation.py`?