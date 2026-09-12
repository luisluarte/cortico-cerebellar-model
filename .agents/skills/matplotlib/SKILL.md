---
name: matplotlib
description: Creating scientific figures, phase portraits, neural time series, raster plots, and publication-quality multi-panel layouts with Matplotlib.
---

# Matplotlib for Scientific & Neural Modeling

## 1. Object-Oriented Multi-Panel Layouts
Always use the Figure and Axes API (`plt.subplots`) instead of stateful `plt.plot`:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex="col")
fig.subplots_adjust(hspace=0.3, wspace=0.25)
```

## 2. Neural Time Series & Error Bands
```python
# Shaded confidence / credible intervals
ax.plot(time, mean_signal, color="#1f77b4", lw=1.5, label="Mean Activation")
ax.fill_between(time, mean_signal - std_signal, mean_signal + std_signal,
                color="#1f77b4", alpha=0.25, label="±1 SD")
```

## 3. Phase Portraits & Vector Fields
```python
# Quiver / Streamplot for 2D dynamical systems
import numpy as np
Y, X = np.mgrid[-2:2:20j, -2:2:20j]
U = -X + np.tanh(X - Y)
V = -Y + np.tanh(X + Y)
ax.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap="viridis", density=1.2)
```

## 4. Dual Y-Axes & Raster Plots
```python
# Dual Y-axis for Drift rate and Threshold
ax2 = ax.twinx()
ax.plot(time, drift_t, "b-", label="Drift v(t)")
ax2.plot(time, threshold_t, "r--", label="Bound a(t)")

# Spike raster plots
ax.eventplot(spike_times_list, lineoffsets=neuron_ids, linelengths=0.8, color="black")
```

## 5. Publication Export
```python
fig.savefig("figure_publication.pdf", dpi=300, bbox_inches="tight", transparent=False)
```
