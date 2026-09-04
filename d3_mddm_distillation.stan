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
  vector[N] soft_pi_1;
  vector[N] soft_mu_1;
  vector[N] soft_sigma_1;
  vector[N] soft_tau_1;
  vector[N] soft_pi_2;
  vector[N] soft_mu_2;
  vector[N] soft_sigma_2;
  vector[N] soft_tau_2;
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
  
  // mDDM Fast Component Parameters
  real<lower=0, upper=1> theta_mix;
  real mu_fast;
  real<lower=0> sigma_fast;
}
model {
  alpha_pos ~ beta(2, 2);
  alpha_neg ~ beta(2, 2);
  beta_v ~ normal(0, 5);
  a ~ normal(1, 1);
  ndt ~ uniform(0, min(RT));
  
  theta_mix ~ beta(1, 9); // Prioritizing deliberate choices, but keeping it open
  mu_fast ~ normal(-1, 1);
  sigma_fast ~ normal(0, 1);
  
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
    
    // 1. Action Choice (Cross-Entropy)
    target += soft_p[t] * log_inv_logit(v_t_full * a) + (1 - soft_p[t]) * log1m_inv_logit(v_t_full * a);
    
    // 2. Mixing Operator Distillation (Cross-Entropy)
    // Distill pi_1 against theta_mix
    target += soft_pi_1[t] * log(theta_mix + 1e-8) + (1 - soft_pi_1[t]) * log(1 - theta_mix + 1e-8);
    
    // 3. Fast Component Distillation (Lognormal Parameters)
    target += -0.5 * square(mu_fast - soft_mu_1[t]) - 0.5 * square(sigma_fast - soft_sigma_1[t]);
    
    // 4. Slow Component Distillation (WFPT Moment Matching against pi_2 moments)
    real v_abs = abs(v_t_full) + 1e-4;
    real expected_rt_wfpt = ndt + (a / (2.0 * v_abs)) * tanh((a * v_abs) / 2.0);
    real v_abs_cubed = v_abs * v_abs * v_abs;
    real var_rt_wfpt = (a * tanh((a * v_abs) / 2.0)) / (2.0 * v_abs_cubed) - square(a) / (4.0 * square(v_abs) * square(cosh((a * v_abs) / 2.0)));
    
    // The surrogate's second component expected moments (Lognormal + tau)
    real E_surrogate_slow = exp(soft_mu_2[t] + square(soft_sigma_2[t]) / 2.0) + soft_tau_2[t];
    real V_surrogate_slow = (exp(square(soft_sigma_2[t])) - 1.0) * exp(2.0 * soft_mu_2[t] + square(soft_sigma_2[t]));
    
    target += -0.5 * square(expected_rt_wfpt - E_surrogate_slow) - 0.5 * square(var_rt_wfpt - V_surrogate_slow);
    
    // 5. Asymmetric Purkinje Update
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
      
      // Calculate true empirical likelihood for the human choice and RT
      // Choice probability
      real p_ch = (ch == c1) ? inv_logit(v_t_full * a) : (1 - inv_logit(v_t_full * a));
      
      // We must bound RT for lognormal to be positive
      // Since tau is absorbed into the WFPT via ndt, the fast Lognormal is assumed to have 0 shift here,
      // or we can just evaluate it natively. 
      real log_lik_fast = lognormal_lpdf(RT[t] | mu_fast, sigma_fast);
      
      // For WFPT, RT must be > ndt. If it's not, wiener_lpdf is -inf. We bound it.
      real log_lik_slow = negative_infinity();
      if (RT[t] > ndt) {
        // wiener_lpdf(RT | a, ndt, beta=0.5, drift)
        // Note: The choice in the DDM is upper vs lower. 
        // If they chose c1, drift is v_t_full. If c2, we flip it.
        // Actually, if we use absolute unconditional RT, we just use abs(v_t_full) for boundary.
        // But the empirical human action was a specific boundary!
        // We evaluate the joint density of choice and RT.
        // Wiener model evaluates first passage time to upper or lower boundary.
        // Upper boundary (Resp=1) has probability, lower (Resp=2).
        if (Resp[t] == 1) {
          log_lik_slow = wiener_lpdf(RT[t] | a, ndt, 0.5, v_t_full);
        } else {
          log_lik_slow = wiener_lpdf(RT[t] | a, ndt, 0.5, -v_t_full);
        }
      }
      
      // The generative probability of choice and RT for the mixture:
      // Fast component: Random guess for choice (prob 0.5) + Lognormal RT
      // Slow component: DDM joint choice and RT
      real log_prob_fast = log(0.5) + log_lik_fast;
      real log_prob_slow = log_lik_slow;
      
      sum_log_lik += log_mix(theta_mix, log_prob_fast, log_prob_slow);
      
      real E_t = Reward[t] - V_ch;
      if (E_t > 0) {
        theta = theta + alpha_pos * E_t * W_exp[:, ch];
      } else {
        theta = theta + alpha_neg * E_t * W_exp[:, ch];
      }
    }
  }
}
