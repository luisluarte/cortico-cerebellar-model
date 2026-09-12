---
name: reservoir-computing-and-dynamics
description: >-
  Mathematical principles and implementation of Reservoir Computing, Echo State Networks (ESN),
  Continuous-Time Recurrent Neural Networks (CTRNN), and nonlinear dynamical systems.
  Covers spectral radius tuning, edge of chaos, Lyapunov exponents, contraction analysis,
  high-dimensional expansion manifolds, and optimal linear/nonlinear readout estimation.
---

# Reservoir Computing and Dynamical Systems

## 1. Mathematical Formulation

### 1.1 Continuous-Time Recurrent Neural Network (CTRNN)
The internal state $z(t) \in \mathbb{R}^n$ evolves according to the nonlinear differential equation:
$$\tau \frac{dz(t)}{dt} = -z(t) + \tanh\left(W_{\text{rec}} z(t) + W_{\text{in}} x(t) + b\right) + \sigma_{\xi} \xi(t)$$
where:
* $\tau$ is the characteristic leak time constant.
* $W_{\text{rec}} \in \mathbb{R}^{n \times n}$ is the internal recurrent weight matrix.
* $W_{\text{in}} \in \mathbb{R}^{n \times k}$ is the input projection matrix.
* $\xi(t)$ is standard Gaussian white noise ($\mathbb{E}[\xi(t)\xi(t')] = \delta(t - t')$).

### 1.2 Discrete Leaky Integrator ESN
$$\tilde{z}_t = \tanh\left(W_{\text{rec}} z_{t-1} + W_{\text{in}} x_t + b\right)$$
$$z_t = (1 - \alpha) z_{t-1} + \alpha \tilde{z}_t$$
where $\alpha \in (0, 1]$ is the leaking rate ($\alpha = \Delta t / \tau$).

---

## 2. Stability, Spectral Radius, and the Edge of Chaos

### 2.1 The Echo State Property (ESP)
The network possesses the Echo State Property if asymptotic states are uniquely determined by the driving input sequence and independent of initial conditions.

1. **Spectral Radius**:
   $$\rho(W_{\text{rec}}) = \max_i |\lambda_i(W_{\text{rec}})|$$
2. **Sufficient Condition for ESP**:
   If the largest singular value $\sigma_{\max}(W_{\text{rec}}) < 1$, the reservoir is strictly contractive.
3. **Edge of Chaos ($\rho \approx 1$)**:
   Operating near the boundary of stability ($\rho \in [0.95, 1.05]$) maximizes:
   * **Information Processing Capacity**: Rich linear and nonlinear memory traces.
   * **Separation Property**: Distinct input trajectories produce orthogonal reservoir states.

### 2.2 Contraction Analysis & Matrix Measures
For nonlinear vector field $\dot{z} = f(z, t)$, contraction requires the generalized matrix measure of the system Jacobian $J(z) = \frac{\partial f}{\partial z}$ to be strictly negative:
$$\mu(J) = \lim_{h \to 0^+} \frac{\|I + h J\| - 1}{h} \le -c < 0$$
Under Euclidean norm: $\mu_2(J) = \lambda_{\max}\left(\frac{J + J^T}{2}\right)$.

### 2.3 Lyapunov Exponents
Quantifies sensitivity to initial conditions:
$$\lambda_{\max} = \lim_{t \to \infty} \lim_{\|\delta z_0\| \to 0} \frac{1}{t} \ln \frac{\|\delta z(t)\|}{\|\delta z_0\|}$$
* $\lambda_{\max} < 0$: Stable fixed point or contractive trajectory.
* $\lambda_{\max} \approx 0$: Critical boundary (edge of chaos, optimal temporal integration).
* $\lambda_{\max} > 0$: Chaotic divergence.

---

## 3. High-Dimensional Expansion & Sparsity

### 3.1 Cover's Theorem on Separability
A complex pattern-classification problem cast nonlinearly into a high-dimensional space is more likely to be linearly separable than in a low-dimensional space:
$$P(\text{separable}) = 2^{1-N} \sum_{i=0}^{n-1} \binom{N-1}{i}$$
for $N$ points in $n$ dimensions. In cerebellar and reservoir models, the unlearned expansion $\phi: \mathbb{R}^k \hookrightarrow \mathbb{R}^n$ ($n \gg k$) transforms nonlinear temporal sequences into linearly decodable static manifolds.

---

## 4. Readout Optimization

### 4.1 Ridge Regression (Tikhonov Regularization)
For linear readout $\hat{y}_t = W_{\text{out}} z_t$:
$$W_{\text{out}} = Y Z^T (Z Z^T + \lambda I)^{-1}$$
where $Z \in \mathbb{R}^{n \times T}$ is the state collection matrix, and $\lambda > 0$ prevents overfitting.

### 4.2 Nonlinear Readouts & L-BFGS
When readout $P_\theta(z_t)$ incorporates thresholding or saturating submersions, optimize parameter vector $\theta$ via Quasi-Newton L-BFGS:
$$\theta^* = \arg\min_\theta \mathcal{L}(y, P_\theta(Z)) + \gamma \|\theta\|_2^2$$
