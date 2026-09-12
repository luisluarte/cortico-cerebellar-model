# Methods

## 1. Teacher-Student Knowledge Distillation and Oracle Representation
To model the complex, chaotic physiological dynamics of the Cortico-Cerebellar network, we employed a Teacher-Student Knowledge Distillation paradigm. This bypasses the mathematical instability of fitting nonlinear ODEs directly to noisy, discrete human behavior.

We defined the complete empirical dataset of $N=100$ participants as **Dataset D**. A Recurrent Neural Network (Teacher) was trained over Dataset D. To ensure robustness, we performed an extensive hyper-parameter optimization grid search for the RNN architecture. Generalization of the Teacher was validated using a Leave-Subject-N-Out (LSNO) cross-validation protocol (10 iterations) over Dataset D, generating a comprehensive performance report.

Following LSNO validation, the optimal RNN was trained over the entirety of Dataset D to form the most complete, perfect continuous representation manifold possible (the "Oracle Teacher"). Its weights were subsequently frozen and saved. The human input sequences were passed through this frozen Oracle to extract trial-by-trial continuous distillation targets: (1) scalar Choice Logits, and (2) 32-dimensional continuous latent states ($\mu_{RNN}$).

## 2. NSGA-II Parameter Discovery
The total population (Dataset D) was split into two distinct, non-overlapping cohorts: **Dataset A** (In-Bag, $N=50$) and **Dataset B** (Out-of-Bag, $N=50$). 

Because physiological ODEs contain massive recurrent feedback loops (e.g., cortico-thalamic reverberations) that shatter auto-differentiation chains, we utilized the Non-Dominated Sorting Genetic Algorithm II (NSGA-II) for initial parameter discovery. Crucially, the NSGA-II algorithm was evolved **exclusively over Dataset A**. It minimized a multi-objective fitness function targeting the Teacher's continuous representations, successfully isolating the stable physiological parameter vector ($\mu_{EB}$) without overfitting to the global population.

## 3. Teacher-Student Reliance Ablation ($\gamma$ Study)
During the distillation process on Dataset A, we implemented a combined optimization metric parameterized by a weighting factor, $\gamma$. This $\gamma$ parameter explicitly controlled the reliance on the Teacher's continuous representations versus the Student's direct generative predictions.

We performed a systematic study to observe how varying the reliance on the Teacher affected the extraction of the biological parameters. We tested the resulting optimized biological models derived from different $\gamma$ quartiles (Q1, Q2, and Q3). Performance was rigorously evaluated using PR-AUC, RT-RMSE, and Negative Log-Likelihood (NLL) to determine the optimal balance between top-down Teacher guidance and bottom-up Student physiological constraints.

## 4. Zero-Shot Out-of-Bag Generalization
To validate the structural and physiological integrity of the parameters discovered over Dataset A ($\mu_{EB}$), we performed a Zero-Shot generalization test on **Dataset B**. 

Using the strictly frozen $\mu_{EB}$ parameters, the Biological Student generated predictions for the $N=50$ unseen participants in Dataset B. We assessed its ability to reconstruct the Teacher's representation manifold on these novel subjects, reporting out-of-bag PR-AUC and RT-RMSE. This confirmed that the NSGA-II optimization did not merely overfit Dataset A, but successfully generalized the underlying biological rules.

## 5. Custom Hierarchical Bayesian MCMC Architecture
Standard Hamiltonian Monte Carlo (HMC) algorithms (e.g., Stan) rely on auto-differentiation, which consistently fails when traversing the chaotic, recurrent algebraic loops of the cortico-cerebellar ODE. 

To conduct rigorous Bayesian inference, we engineered a custom block-wise Metropolis-Hastings (MH) algorithm in C++ (via RcppArmadillo). The hierarchical architecture was defined as:
*   **Population Level ($\mu_{pop}$):** Proposed via a global random-walk and evaluated against specific Bayesian priors and the summed log-likelihood of all subjects.
*   **Subject Level ($Z_{sub}$):** Proposed independently per subject, mapped via the non-centered parameterization $\text{inv\_logit}(\mu_{pop} + \exp(\sigma_{pop}) \odot Z_{sub})$, and constrained by a standard normal prior $N(0,1)$.

To guarantee diagnostic rigor identical to standard HMC software, every Bayesian experiment utilized **3 independent chains** running for **500 iterations** each (with the first 250 discarded as burn-in). Chain convergence and sampling quality were strictly evaluated using trace plots, the Gelman-Rubin statistic ($\hat{R}$), and Effective Sample Size (ESS) metrics.

## 6. The Factorial Evaluation Framework
To isolate the effect of Knowledge Distillation against natively optimized associative baselines (Q-Learning, WSLS), we constructed a $2 \times 2$ factorial evaluation matrix. **Crucially, every experiment was conducted entirely independently on both Dataset A and Dataset B** to explicitly prove that our findings were not artifacts of overfitting. Performance was quantified using Pareto Smoothed Importance Sampling Leave-One-Out cross-validation (PSIS-ELPD).

*   **Experiment 1: Distillation Validation**
    *   *Target:* Teacher RNN Continuous Representations.
    *   *Priors:* Distilled parameters ($\mu_{EB}$) locked as the population mean.
    *   *Objective:* Validate that the biological ODE emulates the Teacher manifold on both A and B.
*   **Experiment 2: The Chaotic Manifold**
    *   *Target:* Raw Empirical Behavior (0/1 Choice, RT).
    *   *Priors:* Fully Unconstrained ($N(0,3)$ prior) for all models.
    *   *Objective:* Prove that unconstrained MCMC optimization of chaotic physiological recurrence fails (diverges) on raw data, whereas heuristic models converge.
*   **Experiment 3: Zero-Shot Transfer**
    *   *Target:* Raw Empirical Behavior.
    *   *Priors:* Bio perfectly locked to $\mu_{EB}$. Q-Learning and WSLS fully unconstrained.
    *   *Objective:* Test whether physiological parameters distilled from an RNN can blindly predict real behavior better than natively optimized empirical heuristics.
*   **Experiment 4: Guided Empirical Bayes Optimization**
    *   *Target:* Raw Empirical Behavior.
    *   *Priors:* Bio guided by an Empirical Bayes prior $N(\mu_{EB}, 0.5)$. Q-Learning and WSLS fully unconstrained.
    *   *Objective:* Test if the physiological model dominates standard reinforcement learning when safely guided to the empirical manifold by the distilled prior, on both the training (A) and holdout (B) datasets.

## 7. Computational Optimization for High-Dimensional Inference
Due to the immense computational load of executing multi-chain MCMC arrays across thousands of ODE simulations, the pipeline was explicitly optimized for maximal remote CPU utilization.
*   **Parallel Multi-Threading:** MCMC chains were structurally decoupled and launched concurrently using the R `future` and `parallel` backends (`mc.cores = detectCores()`), achieving perfect hardware saturation across the remote CPU array.
*   **C++ Vectorization and Shared Memory:** The core biological ODE (`simulate_bio_subject`) and likelihood evaluations were rewritten using highly vectorized Armadillo matrices in C++.
*   **Pre-allocation of Stochastic Matrices:** To defeat MCMC divergence traps caused by unstable Rcpp pseudo-random number generation, the global Brownian noise matrices driving the Purkinje cell drift paths were fully pre-allocated and frozen in memory prior to MCMC initialization, allowing perfectly deterministic topological optimization.
