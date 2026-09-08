
functions {
  real bio_joint_log_lpmf(array[] int y, data matrix X, data vector iti, array[] int subj_id,
                          data matrix W_gen, data matrix W_ach1, data matrix W_ach2, data matrix W_thal, data vector Pi_vec,
                          vector alpha_pc, vector lambda_pc, vector beta_thal, vector kappa_cf,
                          vector alpha_gran, vector beta_gran,
                          vector intercept_c, matrix W_c,
                          data vector RT, vector intercept_rt, matrix W_rt, vector sigma_rt, vector tau_rt);
}
data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y;
  matrix[N, 27] X;
  vector[N] iti;
  vector[N] RT;
  vector[K] min_RT;
  
  matrix[27, 32] W_gen;
  matrix[896, 32] W_ach1;
  matrix[362, 896] W_ach2;
  matrix[32, 362] W_thal;
  vector[27] Pi_vec;

  real<lower=0> prior_mu_alpha_pc;
  real<lower=0> prior_sd_alpha_pc;
  real<lower=0> prior_mu_lambda_pc;
  real<lower=0> prior_sd_lambda_pc;
  real<lower=0> prior_mu_beta_thal;
  real<lower=0> prior_sd_beta_thal;
  real<lower=0> prior_mu_kappa_cf;
  real<lower=0> prior_sd_kappa_cf;
  real<lower=0> prior_mu_alpha_gran;
  real<lower=0> prior_sd_alpha_gran;
  real<lower=0> prior_mu_beta_gran;
  real<lower=0> prior_sd_beta_gran;
}
parameters {
  real alpha_pc_pop_raw;
  real lambda_pc_pop_raw;
  real beta_thal_pop_raw;
  real kappa_cf_pop_raw;
  real alpha_gran_pop_raw;
  real beta_gran_pop_raw;
  
  real<lower=0> sigma_alpha_pc;
  real<lower=0> sigma_lambda_pc;
  real<lower=0> sigma_beta_thal;
  real<lower=0> sigma_kappa_cf;
  real<lower=0> sigma_alpha_gran;
  real<lower=0> sigma_beta_gran;
  
  vector[K] alpha_pc_raw;
  vector[K] lambda_pc_raw;
  vector[K] beta_thal_raw;
  vector[K] kappa_cf_raw;
  vector[K] alpha_gran_raw;
  vector[K] beta_gran_raw;
  
  real intercept_c_pop;
  real<lower=0> sigma_intercept_c;
  vector[K] intercept_c_raw;
  
  vector[32] W_c_pop;
  vector<lower=0>[32] sigma_W_c;
  matrix[32, K] W_c_raw;
  
  real intercept_rt_pop;
  real<lower=0> sigma_intercept_rt;
  vector[K] intercept_rt_raw;
  
  vector[32] W_rt_pop;
  vector<lower=0>[32] sigma_W_rt;
  matrix[32, K] W_rt_raw;
  
  real sigma_rt_pop_raw;
  real<lower=0> sigma_sigma_rt;
  vector[K] sigma_rt_raw;
  
  real tau_rt_pop_raw;
  real<lower=0> sigma_tau_rt;
  vector[K] tau_rt_raw;
}
transformed parameters {
  vector[K] alpha_pc   = 0.5 * inv_logit(alpha_pc_pop_raw   + sigma_alpha_pc   * alpha_pc_raw);
  vector[K] lambda_pc  = inv_logit(lambda_pc_pop_raw  + sigma_lambda_pc  * lambda_pc_raw);
  vector[K] beta_thal  = inv_logit(beta_thal_pop_raw  + sigma_beta_thal  * beta_thal_raw);
  vector[K] kappa_cf   = inv_logit(kappa_cf_pop_raw   + sigma_kappa_cf   * kappa_cf_raw);
  vector[K] alpha_gran = inv_logit(alpha_gran_pop_raw + sigma_alpha_gran * alpha_gran_raw);
  vector[K] beta_gran  = inv_logit(beta_gran_pop_raw  + sigma_beta_gran  * beta_gran_raw);
  
  vector[K] intercept_c = intercept_c_pop + sigma_intercept_c * intercept_c_raw;
  matrix[32, K] W_c;
  for (k in 1:K) { W_c[, k] = W_c_pop + sigma_W_c .* W_c_raw[, k]; }
  
  vector[K] intercept_rt = intercept_rt_pop + sigma_intercept_rt * intercept_rt_raw;
  matrix[32, K] W_rt;
  for (k in 1:K) { W_rt[, k] = W_rt_pop + sigma_W_rt .* W_rt_raw[, k]; }
  
  vector[K] sigma_rt = exp(sigma_rt_pop_raw + sigma_sigma_rt * sigma_rt_raw);
  vector[K] tau_rt = (0.9 * min_RT) .* inv_logit(tau_rt_pop_raw + sigma_tau_rt * tau_rt_raw);
}
model {
  alpha_pc_pop_raw   ~ normal(0, 1);
  lambda_pc_pop_raw  ~ normal(0, 1);
  beta_thal_pop_raw  ~ normal(0, 1);
  kappa_cf_pop_raw   ~ normal(0, 1);
  alpha_gran_pop_raw ~ normal(0, 1);
  beta_gran_pop_raw  ~ normal(0, 1);
  
  sigma_alpha_pc ~ normal(0, 0.5);
  sigma_lambda_pc ~ normal(0, 0.5);
  sigma_beta_thal ~ normal(0, 0.5);
  sigma_kappa_cf ~ normal(0, 0.5);
  sigma_alpha_gran ~ normal(0, 0.5);
  sigma_beta_gran ~ normal(0, 0.5);
  
  alpha_pc_raw ~ std_normal();
  lambda_pc_raw ~ std_normal();
  beta_thal_raw ~ std_normal();
  kappa_cf_raw ~ std_normal();
  alpha_gran_raw ~ std_normal();
  beta_gran_raw ~ std_normal();
  
  intercept_c_pop ~ normal(0, 3);
  sigma_intercept_c ~ normal(0, 1);
  intercept_c_raw ~ std_normal();
  
  W_c_pop ~ double_exponential(0, 0.05); // Bayesian Lasso (L1 Sparsity)
  sigma_W_c ~ normal(0, 1);
  to_vector(W_c_raw) ~ std_normal();
  
  intercept_rt_pop ~ normal(0, 1);
  sigma_intercept_rt ~ normal(0, 0.1);
  intercept_rt_raw ~ std_normal();
  
  W_rt_pop ~ normal(0, 0.1);
  sigma_W_rt ~ normal(0, 0.1);
  to_vector(W_rt_raw) ~ std_normal();
  
  sigma_rt_pop_raw ~ normal(-1, 1);
  sigma_sigma_rt ~ normal(0, 1);
  sigma_rt_raw ~ std_normal();
  
  tau_rt_pop_raw ~ normal(0, 1);
  sigma_tau_rt ~ normal(0, 1);
  tau_rt_raw ~ std_normal();
  
  y ~ bio_joint_log(X, iti, subj_id, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                    alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_gran, beta_gran,
                    intercept_c, W_c, RT, intercept_rt, W_rt, sigma_rt, tau_rt);
}
