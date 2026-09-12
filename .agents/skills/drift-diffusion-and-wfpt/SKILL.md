---
name: drift-diffusion-and-wfpt
description: >-
  Mathematical theory, analytical densities, and simulation workflows for Drift-Diffusion Models (DDM)
  and Wiener First Passage Time (WFPT) distributions.
  Covers boundary hitting times, Navarro-Fuss series switching, dynamic collapsing thresholds,
  linking neural/reservoir dynamics to drift rates and non-decision times, and cognitive decision modeling.
---

# Drift-Diffusion Models & Wiener First Passage Time (WFPT)

## 1. The Continuous Drift-Diffusion Process

The standard Ratcliff Drift-Diffusion Model represents decision-making as a continuous Wiener diffusion process accumulating evidence between two decision thresholds:
$$dX_t = v(t) dt + \sigma dW_t, \quad X_0 = z \cdot a$$
where:
* $v(t) \in \mathbb{R}$ is the drift rate (instantaneous accumulation rate / evidence quality).
* $\sigma$ is diffusion noise (standard scaling convention $\sigma = 1$).
* $a > 0$ is the boundary separation (caution / speed-accuracy tradeoff).
* $z \in (0, 1)$ is the relative starting point bias ($z = 0.5$ indicates unbiased evidence).
* $t_0 \ge 0$ is the non-decision time (sensory encoding + motor execution latency).

An upper boundary hit ($X_t \ge a$) corresponds to choice $A = 1$; a lower boundary hit ($X_t \le 0$) corresponds to choice $A = 0$. Observed Reaction Time is $\text{RT} = t_{\text{hit}} + t_0$.

---

## 2. Analytical WFPT Densities & Series Expansions

The probability density function for first passage through the upper boundary at time $t$ has two equivalent mathematical representations:

### 2.1 Large-Time Expansion (Sine Series)
Converges rapidly for large $t$ ($t > a^2 / \pi$):
$$f(t \mid v, a, z) = \frac{\pi}{a^2} \exp\left(v a (1-z) - \frac{v^2 t}{2}\right) \sum_{k=1}^\infty k \sin(k \pi (1-z)) \exp\left(-\frac{k^2 \pi^2 t}{2 a^2}\right)$$

### 2.2 Small-Time Expansion (Direct Method)
Converges rapidly for small $t$ ($t \le a^2 / \pi$):
$$f(t \mid v, a, z) = \frac{1}{\sqrt{2\pi t^3}} \exp\left(v a (1-z) - \frac{v^2 t}{2}\right) \sum_{k=-\infty}^\infty (2k + 1 - z) a \exp\left(-\frac{((2k + 1 - z)a)^2}{2t}\right)$$

### 2.3 The Navarro-Fuss Adaptive Truncation
To compute the density with strict numerical precision $\varepsilon$ (e.g. $\varepsilon = 10^{-14}$):
* Switch between small-time and large-time expansions at $t^* = \frac{a^2}{\pi}$.
* Truncate the series at $K(\varepsilon)$:
  $$K_{\text{large}} = \left\lceil \frac{\sqrt{-2 \ln(\pi \varepsilon t / a^2)}}{\pi \sqrt{t / a^2}} \right\rceil$$
  $$K_{\text{small}} = \left\lceil \frac{1}{2} + \sqrt{\frac{t}{2a^2}} \sqrt{-\ln(2 \varepsilon \sqrt{2\pi t^3 / a^2})} \right\rceil$$

---

## 3. Dynamic Thresholds & Neural-to-DDM Functors

### 3.1 Collapsing Decision Boundaries
Urgency-gated decisions incorporate time-dependent collapsing thresholds:
$$a(t) = a_0 \cdot \frac{1}{1 + \exp((t - d) / s)} \quad \text{or} \quad a(t) = a_0 \exp(-t / \tau_a)$$
Prevents indefinite deliberation in low-signal environments.

### 3.2 The Embedding Functor ($\Lambda$)
In cortico-cerebellar and neural reservoir architectures, the deterministic internal trajectories map directly into DDM parameters:
$$\Lambda: \mathcal{C} \times \mathcal{X} \to \mathcal{H}_{\text{DDM}}$$
$$\begin{aligned}
v_t &= w_v^T C_t + b_v \\
a_t &= \text{softplus}(w_a^T C_t + b_a) \\
t_{0, t} &= \text{sigmoid}(w_{\text{ndt}}^T X_t + b_{\text{ndt}}) \cdot t_{0, \max}
\end{aligned}$$
This enables end-to-end evaluation of cognitive models directly against empirical RT and choice distributions:
$$p(\text{RT}_i, A_i \mid m) = \text{WFPT}(\text{RT}_i \mid \Lambda(C_i, X_i))$$
