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
    
    // 1. Policy Distillation (Cross-Entropy against unconstrained RNN soft_p)
    target += soft_p[t] * log_inv_logit(v_t_full * a) + (1 - soft_p[t]) * log1m_inv_logit(v_t_full * a);
    
    // 2. Kinematic Empirical Fit (Conditional Wiener Likelihood P(RT | Choice))
    real log_prob_choice;
    real log_wiener;
    
    if (RT[t] > ndt) {
      if (Resp[t] == 1) {
        log_prob_choice = log_inv_logit(v_t_full * a);
        log_wiener = wiener_lpdf(RT[t] | a, ndt, 0.5, v_t_full);
      } else {
        log_prob_choice = log1m_inv_logit(v_t_full * a);
        log_wiener = wiener_lpdf(RT[t] | a, ndt, 0.5, -v_t_full);
      }
      // Add only the conditional RT density to avoid double-counting the choice gradient
      target += (log_wiener - log_prob_choice);
    } else {
      target += negative_infinity();
    }
    
    // 3. Asymmetric Purkinje Update
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      theta = theta + alpha_pos * E_t * W_exp[:, ch];
    } else {
      theta = theta + alpha_neg * E_t * W_exp[:, ch];
    }
  }
}
generated quantities {
  real sum_log_lik = 0;
  {
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
      real V1 = dot_product(theta, W_exp[:, c1]);
      real V2 = dot_product(theta, W_exp[:, c2]);
      real v_t_full = beta_v * (V1 - V2);
      
      if (RT[t] > ndt) {
        if (Resp[t] == 1) {
          sum_log_lik += wiener_lpdf(RT[t] | a, ndt, 0.5, v_t_full);
        } else {
          sum_log_lik += wiener_lpdf(RT[t] | a, ndt, 0.5, -v_t_full);
        }
      }
      
      real E_t = Reward[t] - V_ch;
      if (E_t > 0) {
        theta = theta + alpha_pos * E_t * W_exp[:, ch];
      } else {
        theta = theta + alpha_neg * E_t * W_exp[:, ch];
      }
    }
  }
}
