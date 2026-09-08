
functions {
  real bio_log_lpmf(array[] int y, data matrix X, data vector iti, array[] int subj_id,
                    data matrix W_gen, data matrix W_ach1, data matrix W_ach2, data matrix W_thal, data vector Pi_vec,
                    vector alpha_pc, vector lambda_pc, vector beta_thal, vector kappa_cf,
                    vector alpha_gran, vector beta_gran, vector intercept, matrix W_readout);
}
data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y;
  
  matrix[N, 27] X;
  vector[N] iti;
  
  matrix[27, 32] W_gen;
  matrix[896, 32] W_ach1;
  matrix[362, 896] W_ach2;
  matrix[32, 362] W_thal;
  vector[27] Pi_vec;
  
  real prior_mu_alpha_pc;     real prior_sd_alpha_pc;
  real prior_mu_lambda_pc;    real prior_sd_lambda_pc;
  real prior_mu_beta_thal;    real prior_sd_beta_thal;
  real prior_mu_kappa_cf;     real prior_sd_kappa_cf;
  real prior_mu_alpha_gran;   real prior_sd_alpha_gran;
  real prior_mu_beta_gran;    real prior_sd_beta_gran;
}
parameters {
  real<lower=0> alpha_pc_pop;
  real<lower=0> lambda_pc_pop;
  real<lower=0> beta_thal_pop;
  real<lower=0> kappa_cf_pop;
  real<lower=0> alpha_gran_pop;
  real<lower=0,upper=1> beta_gran_pop;

  real<lower=0> sigma_alpha_pc;
  real<lower=0> sigma_lambda_pc;
  real<lower=0> sigma_beta_thal;
  real<lower=0> sigma_kappa_cf;
  real<lower=0> sigma_alpha_gran;
  real<lower=0> sigma_beta_gran;

  // Centered subject-level parameters
  vector<lower=0>[K] alpha_pc_subj;
  vector<lower=0>[K] lambda_pc_subj;
  vector<lower=0>[K] beta_thal_subj;
  vector<lower=0>[K] kappa_cf_subj;
  vector<lower=0>[K] alpha_gran_subj;
  vector<lower=0,upper=1>[K] beta_gran_subj;
  
  real alpha_pop;
  real<lower=0> sigma_alpha;
  vector[K] alpha_subj;
  
  vector[32] W_readout_pop;
  vector<lower=0>[32] sigma_W_readout;
  matrix[32, K] W_readout_subj;
}
model {
  alpha_pc_pop   ~ normal(prior_mu_alpha_pc, prior_sd_alpha_pc);
  lambda_pc_pop  ~ normal(prior_mu_lambda_pc, prior_sd_lambda_pc);
  beta_thal_pop  ~ normal(prior_mu_beta_thal, prior_sd_beta_thal);
  kappa_cf_pop   ~ normal(prior_mu_kappa_cf, prior_sd_kappa_cf);
  alpha_gran_pop ~ normal(prior_mu_alpha_gran, prior_sd_alpha_gran);
  beta_gran_pop  ~ normal(prior_mu_beta_gran, prior_sd_beta_gran);

  sigma_alpha_pc   ~ normal(0, prior_sd_alpha_pc * 0.5);
  sigma_lambda_pc  ~ normal(0, prior_sd_lambda_pc * 0.5);
  sigma_beta_thal  ~ normal(0, prior_sd_beta_thal * 0.5);
  sigma_kappa_cf   ~ normal(0, prior_sd_kappa_cf * 0.5);
  sigma_alpha_gran ~ normal(0, prior_sd_alpha_gran * 0.5);
  sigma_beta_gran  ~ normal(0, prior_sd_beta_gran * 0.5);

  alpha_pc_subj   ~ normal(alpha_pc_pop, sigma_alpha_pc);
  lambda_pc_subj  ~ normal(lambda_pc_pop, sigma_lambda_pc);
  beta_thal_subj  ~ normal(beta_thal_pop, sigma_beta_thal);
  kappa_cf_subj   ~ normal(kappa_cf_pop, sigma_kappa_cf);
  alpha_gran_subj ~ normal(alpha_gran_pop, sigma_alpha_gran);
  beta_gran_subj  ~ normal(beta_gran_pop, sigma_beta_gran);

  alpha_pop ~ normal(0, 3);
  sigma_alpha ~ normal(0, 1);
  alpha_subj ~ normal(alpha_pop, sigma_alpha);
  
  W_readout_pop ~ normal(0, 3);
  sigma_W_readout ~ normal(0, 1);
  for (k in 1:K) {
    W_readout_subj[, k] ~ normal(W_readout_pop, sigma_W_readout);
  }

  y ~ bio_log(X, iti, subj_id, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
              alpha_pc_subj, lambda_pc_subj, beta_thal_subj, kappa_cf_subj,
              alpha_gran_subj, beta_gran_subj, alpha_subj, W_readout_subj);
}
