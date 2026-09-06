data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1,upper=S> subj;
  array[N] int<lower=0,upper=1> y_switch;
  vector[N] y_rt;
  vector[N] ITI;
  
  // Model 1 Parameters
  real alpha_pc;
  real lambda_pc;
  real beta_thal;
  real kappa_cf;
  real alpha_granule;
  real beta_granule;
  
  // Fixed Anatomy Matrices
  matrix[27, 32] W_gen;
  matrix[896, 32] W_ach1;
  matrix[362, 896] W_ach2;
  matrix[32, 362] W_thal;
  vector[27] Pi_vec;
  
  // Inputs
  matrix[N, 27] X;
  matrix[N, 896] W_purk_noise; // Precomputed noise to keep chains perfectly identical
  
  // RNN Predictions
  vector[N] rnn_p_logits;
  matrix[N, 2] rnn_mu;
  matrix[N, 2] rnn_sigma;
  matrix[N, 2] rnn_tau;
  matrix[N, 2] rnn_pi;
}

transformed data {
  matrix[N, 32] mu_history;
  vector[32] mu = rep_vector(0.0, 32);
  vector[896] Z = rep_vector(0.0, 896);
  vector[896] W_purk = rep_vector(0.0, 896);
  vector[362] D = rep_vector(0.0, 362);
  
  // 1. Fully static C++ Biological Euler Simulation
  for (t in 1:N) {
    if (ITI[t] > 0.0) {
      Z = rep_vector(0.0, 896);
      W_purk += W_purk_noise[t]';
    }
    
    vector[27] I_t = X[t]';
    vector[27] I_hat = W_gen * mu;
    vector[27] eps = Pi_vec .* (I_t - I_hat);
    
    mu = mu + alpha_pc * (W_gen' * eps - (lambda_pc * ITI[t]) * mu + beta_thal * (W_thal * D));
    vector[896] G = W_ach1 * mu;
    Z = (1.0 - beta_granule) * Z + alpha_granule * G;
    
    real eps_mag = mean(fabs(eps));
    W_purk = W_purk - kappa_cf * (eps_mag * Z);
    D = W_ach2 * (G .* W_purk);
    
    mu_history[t] = mu';
  }
  
  // 2. Analytic Ridge Regression 
  matrix[32, 32] XX = crossprod(mu_history) + diag_matrix(rep_vector(0.01, 32));
  matrix[32, 32] XX_inv = inverse_spd(XX);
  
  vector[32] W_policy = XX_inv * (mu_history' * rnn_p_logits);
  matrix[32, 2] W_mu = XX_inv * (mu_history' * rnn_mu);
  
  vector[N] bio_p_logits = mu_history * W_policy;
  matrix[N, 2] bio_mu = mu_history * W_mu;
}

parameters {
  real mu_gamma_raw;
  real<lower=0> sigma_gamma_raw;
  vector[S] gamma_raw_subj;
}

transformed parameters {
  vector[S] gamma = inv_logit(mu_gamma_raw + sigma_gamma_raw * gamma_raw_subj);
}

model {
  // Tight hierarchical priors centered on Pareto Model #1 (gamma ~ 0.216 -> logit ~ -1.28)
  mu_gamma_raw ~ normal(-1.28, 0.5); 
  sigma_gamma_raw ~ normal(0, 0.5);
  gamma_raw_subj ~ std_normal();
  
  for (i in 1:N) {
    real g = gamma[subj[i]];
    
    // Mix Policy
    real blend_p_logit = (1.0 - g) * rnn_p_logits[i] + g * bio_p_logits[i];
    target += bernoulli_logit_lpmf(y_switch[i] | blend_p_logit);
    
    // Mix Kinematics (MDN)
    vector[2] log_p_k;
    for (k in 1:2) {
       real blend_mu_k = (1.0 - g) * rnn_mu[i, k] + g * bio_mu[i, k];
       real shifted_rt = y_rt[i] - rnn_tau[i, k];
       if (shifted_rt <= 1e-6) {
          log_p_k[k] = -1000.0;
       } else {
          real log_y = log(shifted_rt);
          log_p_k[k] = log(rnn_pi[i, k]) - log_y - 0.5*log(2*pi()) - log(rnn_sigma[i, k]) - 0.5 * square((log_y - blend_mu_k) / rnn_sigma[i, k]);
       }
    }
    target += log_sum_exp(log_p_k);
  }
}

generated quantities {
  // Extract pointwise log-likelihood for ELPD / LOO-CV Model Comparison
  vector[N] log_lik;
  for (i in 1:N) {
    real g = gamma[subj[i]];
    real blend_p_logit = (1.0 - g) * rnn_p_logits[i] + g * bio_p_logits[i];
    real ll_choice = bernoulli_logit_lpmf(y_switch[i] | blend_p_logit);
    
    vector[2] log_p_k;
    for (k in 1:2) {
       real blend_mu_k = (1.0 - g) * rnn_mu[i, k] + g * bio_mu[i, k];
       real shifted_rt = y_rt[i] - rnn_tau[i, k];
       if (shifted_rt <= 1e-6) {
          log_p_k[k] = -1000.0;
       } else {
          real log_y = log(shifted_rt);
          log_p_k[k] = log(rnn_pi[i, k]) - log_y - 0.5*log(2*pi()) - log(rnn_sigma[i, k]) - 0.5 * square((log_y - blend_mu_k) / rnn_sigma[i, k]);
       }
    }
    log_lik[i] = ll_choice + log_sum_exp(log_p_k);
  }
}
