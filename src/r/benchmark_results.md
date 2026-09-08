# Benchmark Results for Hierarchical BPTT Models (N=5)
- **Model 1: Baseline HMC (NCP, Diagonal Metric)**: Success | 270.46 seconds (0.07 iters/sec)
- **Model 2: Baseline HMC (NCP, Dense Metric)**: Success | 177.48 seconds (0.11 iters/sec)
- **Model 3: HMC Centered Parameterization (CP, Diagonal Metric)**: Success | 214.22 seconds (0.09 iters/sec)
- **Model 4: Laplace Approximation (Optimize + Hessian)**: Failed: 'jacobian' argument to optimize and laplace must match!
laplace was called with jacobian=TRUE
optimize was run with jacobian=FALSE | 32.85 seconds (NA iters/sec)
- **Model 5: Variational Inference (ADVI)**: Success | 28.16 seconds (0.71 iters/sec)
