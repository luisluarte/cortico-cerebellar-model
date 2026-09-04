data {
  int<lower=1> N;
  int<lower=1> N_subj;
  array[N] int<lower=1> subj;
  array[N] int<lower=2, upper=8> Bd1;
  array[N] int<lower=2, upper=8> Bd2;
  array[N] int<lower=1, upper=2> Resp;
  array[N] real Reward;
  array[N] real RT;
  
  vector[N] soft_p;
  vector[N] soft_mu;
  vector[N] soft_sigma;
  vector[N] soft_tau;
}
transformed data {
  int n_dim = 50;
  matrix[n_dim, 8] W_exp;
  for (i in 1:n_dim) {
    for (j in 1:8) {
      W_exp[i, j] = normal_rng(0, 1 / sqrt(50.0));
    }
  }
}
parameters {
  real<lower=0, upper=1> alpha_pos;
  real<lower=0, upper=1> alpha_neg;
  real<lower=0, upper=20> beta_v;
  real<lower=0, upper=5> a;
  real<lower=0, upper=min(RT)> ndt;
}
model {
  alpha_pos ~ beta(2, 2);
  alpha_neg ~ beta(2, 2);
  beta_v ~ normal(0, 5);
  a ~ normal(1, 1);
  ndt ~ uniform(0, min(RT));
  
  vector[n_dim] theta;
  int current_subj = -1;
  
  for (t in 1:N) {
    if (subj[t] != current_subj) {
      theta = rep_vector(0.0, n_dim);
      current_subj = subj[t];
    }
    
    int c1 = Bd1[t];
    int c2 = Bd2[t];
    int ch = Resp[t] == 1 ? c1 : c2;
    int unch = Resp[t] == 1 ? c2 : c1;
    
    real V_ch = dot_product(theta, W_exp[:, ch]);
    real V_unch = dot_product(theta, W_exp[:, unch]);
    real V1 = dot_product(theta, W_exp[:, c1]);
    real V2 = dot_product(theta, W_exp[:, c2]);
    real v_t_full = beta_v * (V1 - V2);
    
    target += soft_p[t] * log_inv_logit(v_t_full * a) + (1 - soft_p[t]) * log1m_inv_logit(v_t_full * a);
    
    real mu_hat_t = exp(soft_mu[t] + square(soft_sigma[t]) / 2.0) + soft_tau[t];
    real sigma_sq_hat_t = (exp(square(soft_sigma[t])) - 1.0) * exp(2.0 * soft_mu[t] + square(soft_sigma[t]));
    
    real v_abs = abs(v_t_full) + 1e-4;
    real expected_rt_wfpt = ndt + (a / (2.0 * v_abs)) * tanh((a * v_abs) / 2.0);
    real v_abs_cubed = v_abs * v_abs * v_abs;
    real var_rt_wfpt = (a * tanh((a * v_abs) / 2.0)) / (2.0 * v_abs_cubed) - square(a) / (4.0 * square(v_abs) * square(cosh((a * v_abs) / 2.0)));
    
    real scale_mu = 1.0;
    real scale_var = 1.0;
    target += -0.5 * square((expected_rt_wfpt - mu_hat_t) / scale_mu) - 0.5 * square((var_rt_wfpt - sigma_sq_hat_t) / scale_var);
    
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      theta = theta + alpha_pos * E_t * W_exp[:, ch];
    } else {
      theta = theta + alpha_neg * E_t * W_exp[:, ch];
    }
  }
}
