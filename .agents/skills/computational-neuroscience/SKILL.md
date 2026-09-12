---
name: computational-neuroscience
description: >-
  Expert principles, mathematical formulations, and simulation workflows for computational neuroscience.
  Covers spiking neuron models (LIF, Izhikevich, Hodgkin-Huxley), rate networks (Wilson-Cowan),
  cerebellar microcircuit topology (Marr-Albus-Ito, granule expansion, Purkinje compression, olivary climbing fiber error signals),
  cortico-cerebellar predictive coding loops, and simulation tools (Brian2, NEST, Arbor).
---

# Computational Neuroscience

## 1. Biophysical & Spiking Neuron Models

### 1.1 Leaky Integrate-and-Fire (LIF)
The fundamental linear subthreshold integration equation with a hard reset:
$$\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{\text{rest}}) + R_m I(t)$$
Upon threshold crossing $V(t) \ge V_{\text{th}}$:
1. Emit action potential spike at $t = t_{\text{spike}}$.
2. Instantly reset $V(t) \leftarrow V_{\text{reset}}$.
3. Enforce refractory period $\tau_{\text{ref}}$ during which $\frac{dV}{dt} = 0$.

### 1.2 Izhikevich 2D Quadratic Model
Combines biological plausibility of Hodgkin-Huxley with the computational efficiency of integrate-and-fire:
$$\frac{dv}{dt} = 0.04 v^2 + 5v + 140 - u + I$$
$$\frac{du}{dt} = a(bv - u)$$
Auxiliary reset condition: if $v \ge 30\text{ mV}$, then $v \leftarrow c$ and $u \leftarrow u + d$.
* Regular spiking (RS): $a=0.02, b=0.2, c=-65, d=8$
* Fast spiking (FS, interneurons): $a=0.1, b=0.2, c=-65, d=2$
* Bursting (e.g. cerebellar granule/deep nuclei): $a=0.02, b=0.2, c=-50, d=2$

### 1.3 Hodgkin-Huxley 4D Conductance Dynamics
Full biophysical voltage-gated ion channel description:
$$C_m \frac{dV}{dt} = I_{\text{inj}} - \bar{g}_{\text{Na}} m^3 h (V - E_{\text{Na}}) - \bar{g}_{\text{K}} n^4 (V - E_{\text{K}}) - g_L (V - E_L)$$
$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x, \quad x \in \{m, h, n\}$$

---

## 2. Neural Population & Rate Dynamics

### 2.1 Wilson-Cowan Excitatory-Inhibitory (E-I) System
Models the mean firing rates of coupled excitatory ($E$) and inhibitory ($I$) populations:
$$\tau_E \frac{dE}{dt} = -E + S_E(w_{EE} E - w_{EI} I + P_E)$$
$$\tau_I \frac{dI}{dt} = -I + S_I(w_{IE} E - w_{II} I + P_I)$$
where $S_k(x) = \frac{1}{1 + \exp(-\beta_k(x - \theta_k))}$ is the sigmoid activation functional.

---

## 3. Cerebellar Microcircuit Topology & Predictive Coding

### 3.1 The Canonical Marr-Albus-Ito Architecture
The cerebellum implements an adaptive expansion-compression predictive filter:
1. **Input & Divergent Expansion**:
   * Mossy Fibers (MF) convey multimodal sensory, motor, and cognitive context states $X_t \in \mathcal{X} \subseteq \mathbb{R}^k$.
   * Granule Cells (GC) expand this input into a high-dimensional space $\mathcal{Z} \subseteq \mathbb{R}^n$ ($n \gg k$, biologically $\approx 10^{11}$ cells).
   * Golgi Cells provide recurrent divisive normalization and inhibition, enforcing extreme sparsity (active fraction $\sim 1\% - 5\%$).
2. **Submersion & Readout**:
   * Parallel Fibers (PF, axons of GC) form hundreds of thousands of synapses onto each Purkinje Cell (PC).
   * The Purkinje cell layer compresses high-dimensional activity into low-dimensional inhibitory readouts $C_t = P_\theta(\phi(X_t)) \in \mathcal{C} \subseteq \mathbb{R}^d$.
3. **Climbing Fiber Error Guidance**:
   * Inferior Olive (IO) sends Climbing Fibers (CF) to Purkinje cells with a 1:1 topological innervation.
   * Climbing fiber spikes convey signed teaching/error signals $E_t = R_t - V(C_t, A_t)$.
   * Concomitant PF + CF activation induces Long-Term Depression (LTD) at PF-PC synapses; absence of CF signals allows slow Long-Term Potentiation (LTP).
4. **Deep Cerebellar Nuclei (DCN)**:
   * Receive excitatory MF collaterals and sculpted inhibitory PC output, projecting forward corrections to the cerebral cortex and motor effectors.

### 3.2 Cortico-Cerebellar Predictive Coding Loops
* Cerebral cortex generates state predictions and action policies.
* Cerebellar internal models compute forward dynamics: predicting sensory consequences before slow peripheral feedback arrives.
* Discrepancy between predicted and actual outcomes forms the climbing fiber error $E_t$, updating cerebellar compression matrices without corrupting cortical representations.
