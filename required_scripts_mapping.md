# Manuscript Results to Scripts Mapping

## 1. Multi-objective distillation of biological parameters
* **RNN Teacher Validation (ROC-AUC, PR-AUC, Accuracy, CRPS calibration):** 
  * `src/optimization/pure_mdn_lnso_crps.R`
* **NSGA-II Parameter Discovery (4D Pareto front hypervolume, WSLS vs Q-learning, autonomy decay rate $\beta_{log}$):** 
  * `compute_hypercost_hv.R`
  * `bootstrap_slopes_hypercost.R`
  * `master_4d_hypercost.R`

## 2. Non-Markovian state maintenance via cortico-cerebellar feedback
* **Memory Capacity (MC) and Sequential Thalamic Feedback Ablation (1% increments):** 
  * `run_hallucination_thal_ablation.R`
  * `generate_true_plots.py` (Linear mixed-effects modeling & plotting of Memory Capacity collapse)
  * `generate_split_ablations.py`

## 3. Geometric approximation of optimal Bayesian inference
* **Bayesian Leave-One-Out Cross-Validation (PSIS-LOO, ELPD metrics, Pareto k warnings):** 
  * `get_all_stats.R`
  * `get_all_stats2.R`
  * `review_biodist.R`

## 4. Directional Flow of Information and Representation Stability
* **Granger Causality (Directional Flow between Cortex and Cerebellum):** 
  * `run_granger_fstats.R`
* **Raw Encoding Capacity & Stabilized Readout Efficiency (Value, State, Volatility):** 
  * `run_readout_efficiency.R`
  * `stat_eff_final.R`

## 5. Synthetic Thalamic ablation induced representation collapse and perseveration
* **Encoding Efficiency under $\text{Lesion}_{\text{Thal}}$ and $\text{Lesion}_{\kappa}$:** 
  * `run_lesion_stats.R`
  * `run_lesion_efficiency.R`
* **Policy Perseverance (WSLS Beta posterior differential entropy):** 
  * `run_perseveration.R`
* **Trial-by-trial online readout (Delta Rule online Choice NLL):** 
  * `run_true_rolling_ridge.R`
  * `run_rolling_ridge.R`
