data {
  int<lower=1> N;            // Total number of trials across all N=3 subjects
  int<lower=1> K;            // Number of subjects (3)
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y; // Switch choice (1 or 0)
  
  matrix[N, 27] X;           // Input vectors
  vector[N] iti;             // ITI values
  
  // Fixed Network Weights (Data)
  matrix[27, 32] W_gen;
  matrix[896, 32] W_ach1;
  matrix[362, 896] W_ach2;
  matrix[32, 362] W_thal;
  vector[27] Pi_vec;
  
  // Informative Prior Hyperparameters (from NSGA-II)
  // We use tighter variance for the population priors as requested
  real prior_mu_alpha_pc;     real prior_sd_alpha_pc;
  real prior_mu_lambda_pc;    real prior_sd_lambda_pc;
  real prior_mu_beta_thal;    real prior_sd_beta_thal;
  real prior_mu_kappa_cf;     real prior_sd_kappa_cf;
  real prior_mu_alpha_gran;   real prior_sd_alpha_gran;
  real prior_mu_beta_gran;    real prior_sd_beta_gran;
  // sigma2_diff is omitted as we compute the deterministic Mean Trajectory
}
parameters {
  // Biological Population Means (Strictly positive)
  real<lower=0> alpha_pc_pop;
  real<lower=0> lambda_pc_pop;
  real<lower=0> beta_thal_pop;
  real<lower=0> kappa_cf_pop;
  real<lower=0> alpha_gran_pop;
  real<lower=0> beta_gran_pop;

  // Biological Population SDs
  real<lower=0> sigma_alpha_pc;
  real<lower=0> sigma_lambda_pc;
  real<lower=0> sigma_beta_thal;
  real<lower=0> sigma_kappa_cf;
  real<lower=0> sigma_alpha_gran;
  real<lower=0> sigma_beta_gran;

  // Biological Subject-Level Raw Deviations
  vector[K] alpha_pc_raw;
  vector[K] lambda_pc_raw;
  vector[K] beta_thal_raw;
  vector[K] kappa_cf_raw;
  vector[K] alpha_gran_raw;
  vector[K] beta_gran_raw;
  
  // Readout Parameters
  real alpha_pop; // population intercept
  real<lower=0> sigma_alpha;
  vector[K] alpha_raw;
  
  vector[32] W_readout_pop;
  vector<lower=0>[32] sigma_W_readout;
  matrix[32, K] W_readout_raw;
}
transformed parameters {
  // Map Raw to Subject-Level Parameters
  vector[K] alpha_pc_subj   = alpha_pc_pop   + sigma_alpha_pc   * alpha_pc_raw;
  vector[K] lambda_pc_subj  = lambda_pc_pop  + sigma_lambda_pc  * lambda_pc_raw;
  vector[K] beta_thal_subj  = beta_thal_pop  + sigma_beta_thal  * beta_thal_raw;
  vector[K] kappa_cf_subj   = kappa_cf_pop   + sigma_kappa_cf   * kappa_cf_raw;
  vector[K] alpha_gran_subj = alpha_gran_pop + sigma_alpha_gran * alpha_gran_raw;
  vector[K] beta_gran_subj  = beta_gran_pop  + sigma_beta_gran  * beta_gran_raw;
  
  // Readout subject parameters
  vector[K] alpha_subj = alpha_pop + sigma_alpha * alpha_raw;
  matrix[32, K] W_readout_subj;
  for (k in 1:K) {
    W_readout_subj[, k] = W_readout_pop + sigma_W_readout .* W_readout_raw[, k];
  }
}
model {
  // Priors for Biological Population Means (Informative from NSGA-II)
  alpha_pc_pop   ~ normal(prior_mu_alpha_pc, prior_sd_alpha_pc);
  lambda_pc_pop  ~ normal(prior_mu_lambda_pc, prior_sd_lambda_pc);
  beta_thal_pop  ~ normal(prior_mu_beta_thal, prior_sd_beta_thal);
  kappa_cf_pop   ~ normal(prior_mu_kappa_cf, prior_sd_kappa_cf);
  alpha_gran_pop ~ normal(prior_mu_alpha_gran, prior_sd_alpha_gran);
  beta_gran_pop  ~ normal(prior_mu_beta_gran, prior_sd_beta_gran);

  // Priors for Biological SDs (Half-Normal, tighter)
  sigma_alpha_pc   ~ normal(0, prior_sd_alpha_pc * 0.5);
  sigma_lambda_pc  ~ normal(0, prior_sd_lambda_pc * 0.5);
  sigma_beta_thal  ~ normal(0, prior_sd_beta_thal * 0.5);
  sigma_kappa_cf   ~ normal(0, prior_sd_kappa_cf * 0.5);
  sigma_alpha_gran ~ normal(0, prior_sd_alpha_gran * 0.5);
  sigma_beta_gran  ~ normal(0, prior_sd_beta_gran * 0.5);

  // Raw deviations
  alpha_pc_raw   ~ std_normal();
  lambda_pc_raw  ~ std_normal();
  beta_thal_raw  ~ std_normal();
  kappa_cf_raw   ~ std_normal();
  alpha_gran_raw ~ std_normal();
  beta_gran_raw  ~ std_normal();

  // Readout Priors (L2 Regularization)
  alpha_pop ~ normal(0, 3);
  sigma_alpha ~ normal(0, 1);
  alpha_raw ~ std_normal();
  
  W_readout_pop ~ normal(0, 3);
  sigma_W_readout ~ normal(0, 1);
  to_vector(W_readout_raw) ~ std_normal();

  // The Mechanistic Integration Loop
  vector[N] logits;
  vector[32] mu;
  vector[896] Z;
  vector[896] W_purk;
  vector[362] D;
  
  int current_subj = -1;
  
  // Pre-transpose for faster multiplication if needed, but matrices are fixed.
  matrix[32, 27] W_gen_t = W_gen';

  for (t in 1:N) {
    // Reset state if new subject
    if (subj_id[t] != current_subj) {
      mu = rep_vector(0.0, 32);
      Z = rep_vector(0.0, 896);
      W_purk = rep_vector(0.0, 896);
      D = rep_vector(0.0, 362);
      current_subj = subj_id[t];
    }
    
    // In sim_bio.cpp: W_purk adds random noise if ITI > 0.
    // For deterministic HMC mean-trajectory, the expected drift is 0.
    // if (iti[t] > 0) { // W_purk += randn() * sqrt(sigma2 * iti) } => ignored.
    
    vector[27] I_t = X[t]';
    vector[27] I_hat = W_gen * mu;
    vector[27] eps = Pi_vec .* (I_t - I_hat);
    
    // Bio Subject Parameters
    int k = current_subj;
    real p_alpha_pc   = alpha_pc_subj[k];
    real p_lambda_pc  = lambda_pc_subj[k];
    real p_beta_thal  = beta_thal_subj[k];
    real p_kappa_cf   = kappa_cf_subj[k];
    real p_alpha_gran = alpha_gran_subj[k];
    real p_beta_gran  = beta_gran_subj[k];
    
    mu = mu + p_alpha_pc * ((W_gen_t * eps) - (p_lambda_pc * iti[t]) * mu + p_beta_thal * (W_thal * D));
    
    vector[896] G = W_ach1 * mu;
    Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
    
    real eps_mag = mean(abs(eps));
    W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
    
    D = W_ach2 * (G .* W_purk);
    
    // Save readout logit for the current trial
    logits[t] = alpha_subj[k] + dot_product(mu, col(W_readout_subj, k));
  }
  
  y ~ bernoulli_logit(logits);
}
