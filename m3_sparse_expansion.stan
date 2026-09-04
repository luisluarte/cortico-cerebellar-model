data {
  int<lower=1> N;
  int<lower=1> N_subj;
  array[N] int<lower=1> subj;
  array[N] int<lower=2, upper=8> Bd1;
  array[N] int<lower=2, upper=8> Bd2;
  array[N] int<lower=1, upper=2> Resp;
  array[N] real Reward;
  array[N] real RT;
  array[N] real ITI;
}
transformed data {
  int n_dim = 200; // Expanded latent manifold
  matrix[n_dim, 8] W_exp;
  for (i in 1:n_dim) {
    for (j in 1:8) {
      // 80% sparsity
      if (uniform_rng(0, 1) > 0.8) {
        W_exp[i, j] = normal_rng(0, 1 / sqrt(200.0));
      } else {
        W_exp[i, j] = 0.0;
      }
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
  // Priors
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
    
    // M3: Unconstrained readout over sparse expanded reservoir
    real V_ch = dot_product(theta, W_exp[:, ch]);
    real V_unch = dot_product(theta, W_exp[:, unch]);
    
    real v_t = beta_v * (V_ch - V_unch);
    
    // Log likelihood
    if (Resp[t] == 1) {
      RT[t] ~ wiener(a, ndt, 0.5, v_t);
    } else {
      RT[t] ~ wiener(a, ndt, 0.5, -v_t);
    }
    
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      theta = theta + alpha_pos * E_t * W_exp[:, ch];
    } else {
      theta = theta + alpha_neg * E_t * W_exp[:, ch];
    }
  }
}
generated quantities {
  vector[N] log_lik;
  vector[n_dim] theta_gen;
  int current_subj_gen = -1;
  
  for (t in 1:N) {
    if (subj[t] != current_subj_gen) {
      theta_gen = rep_vector(0.0, n_dim);
      current_subj_gen = subj[t];
    }
    
    int c1 = Bd1[t];
    int c2 = Bd2[t];
    int ch = Resp[t] == 1 ? c1 : c2;
    int unch = Resp[t] == 1 ? c2 : c1;
    
    real V1 = dot_product(theta_gen, W_exp[:, c1]);
    real V2 = dot_product(theta_gen, W_exp[:, c2]);
    real v_t = beta_v * (V1 - V2);
    
    if (Resp[t] == 1) {
      log_lik[t] = wiener_lpdf(RT[t] | a, ndt, 0.5, v_t);
    } else {
      log_lik[t] = wiener_lpdf(RT[t] | a, ndt, 0.5, -v_t);
    }
    
    real V_ch = dot_product(theta_gen, W_exp[:, ch]);
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      theta_gen = theta_gen + alpha_pos * E_t * W_exp[:, ch];
    } else {
      theta_gen = theta_gen + alpha_neg * E_t * W_exp[:, ch];
    }
  }
}
