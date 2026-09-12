# Cortico-Cerebellar Dynamics Model

This repository contains the official code, simulation environments, and data analysis scripts for our publication on the continuous dynamical models of the cortico-cerebellar loop and its impact on adaptive behavior.

## Overview

Unlike standard cognitive modeling that fits static heuristics (e.g., Win-Stay/Lose-Shift) to behavioral data, this project introduces a **closed-loop generative framework**. We model the exact working memory state, cortical integration rate, and thalamic feedback rigidity of human participants using adaptive Markov Chain Monte Carlo (MCMC) and NSGA-II optimized Recurrent Neural Networks. 

### Key Contributions
1. **The Compensatory Manifold**: We demonstrate that human biological parameters (such as cortical learning rate and thalamic feedback) form non-linear compensatory manifolds (Spearman $\rho = 0.95$).
2. **Phase Transitions in Confidence**: We show that decision confidence (measured as the L2-norm of the cortical state) undergoes a sharp dynamical phase transition as thalamic rigidity increases.
3. **Ecological Rationality**: We provide C++ environments simulating Restless Bandit tasks to demonstrate that the optimal thalamic feedback rigidity fundamentally depends on the environmental volatility.

---

## Repository Structure

The repository adheres to FAIR Open Science principles. Data, computational engines, and analytical scripts are decoupled for maximum reproducibility.

- **`data/`**: Contains the empirical human datasets, notably `behavioral_compilate.csv`.
- **`src/r/`**: Contains the custom C++ Markov Chain samplers (`bio_mcmc_*.cpp`), the generative continuous simulators (`sim_bio_probes.cpp`, `sim_agent.cpp`), and all analytical R scripts mapping the latent spaces to empirical behavior.
- **`results/`**: Output directory containing optimal parameter matrices (`factorial_fixed_10k_A.rds`) and all figures generated for the manuscript.
- **`manuscript/`**: The LaTeX source code and bibliography for the publication.

---

## Installation & Dependencies

To reproduce the findings, ensure you have **R (>= 4.0.0)** and a functioning **C++ compiler** installed on your system (e.g., `Rtools` for Windows, `gcc` for Linux/macOS) to compile the Rcpp Armadillo engines.

To install all exact R dependencies and PyTorch bindings, simply run:
```bash
Rscript requirements.R
```

---

## Reproduction Guide (Figure Generation)

Each manuscript figure corresponds directly to a single, executable analytical script. Ensure your working directory is the repository root, then run the corresponding script.

### 1. Representation & The Latent State
- **Figure 5 (Latent Decoding):** `Rscript decode_latent_traces.R`
- **Figure 18 (Representation Collapse):** `Rscript correlate_representation.R`

### 2. Autonomous Attractor Dynamics
- **Figure 6 (Sensory Deprivation Hallucinations):** `Rscript run_hallucination_beta_thal.R`
- **Figure 7 (Lesioning the Thalamic Loop):** `Rscript run_lesion_beta_thal.R`
- **Figure 17 (Internal Confidence Phase Transition):** `Rscript measure_confidence.R`

### 3. Empirical Behavioral Topography
- **Figure 12 (3D Behavioral Landscape):** `Rscript plot_behavioral_landscape.R`
- **Figure 13 (The Behavioral U-Shape):** `Rscript plot_binned_markovian.R`
- **The Compensatory Manifold (Beta vs Alpha):** `Rscript test_params.R`

### 4. Causal & Generative Simulations
- **Figure 14 (Simulated Generative Agent):** `Rscript run_agent_simulation.R`
- **Figure 15 (Restless Bandit Volatility):** `Rscript run_restless_sweep.R`

---

## Authors & Citation
*(Placeholder for Final Authorship and Zenodo DOI)*

When referring to this code, please cite the corresponding paper. A `CITATION.cff` will be provided upon publication.
