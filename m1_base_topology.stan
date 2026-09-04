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
  // Priors (non-hierarchical, shared across N=30)
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
    
    // Decay compression weights? The base_topology doesn't explicitly specify decay, 
    // but the data dict mentions ITI is critical for memory decay. 
    // Let's omit ITI decay for M1 to keep it purely mapping to the base topology, 
    // or add a simple decay parameter. Let's omit for strict adherence to Omega.
    
    int c1 = Bd1[t];
    int c2 = Bd2[t];
    int ch = Resp[t] == 1 ? c1 : c2;
    int unch = Resp[t] == 1 ? c2 : c1;
    
    real V_ch = dot_product(theta, W_exp[:, ch]);
    real V_unch = dot_product(theta, W_exp[:, unch]);
    
    // DDM Embedding
    real drift = beta_v * (V_ch - V_unch);
    
    // Wiener First Passage Time
    // Resp[t] == 1 means bounded at 'a' (chose Bd1)
    // Resp[t] == 2 means bounded at '0' (chose Bd2)
    // Actually, let's map chosen/unchosen to upper/lower bound? 
    // Or Bd1 to upper bound, Bd2 to lower bound.
    // If drift = beta * (V1 - V2), then upper bound (1) is Bd1, lower bound (0) is Bd2.
    real V1 = dot_product(theta, W_exp[:, c1]);
    real V2 = dot_product(theta, W_exp[:, c2]);
    real v_t = beta_v * (V1 - V2);
    
    // Stan's wiener_lpdf(Y | a, ndt, bias, drift)
    // RT must be >= ndt
    // If ch == Bd1 (Resp=1), hitting upper bound, bias is typically 0.5.
    // However, Stan's wiener takes a single RT and we need to separate upper vs lower hit.
    // We can use the standard trick: if lower bound is hit, RT is negative? No, Stan's wiener doesn't do that.
    // Wait, Stan's wiener_lpdf models the upper boundary! To model the lower boundary, we flip the drift and bias.
    if (Resp[t] == 1) {
      RT[t] ~ wiener(a, ndt, 0.5, v_t);
    } else {
      RT[t] ~ wiener(a, ndt, 0.5, -v_t);
    }
    
    // Asymmetric Vector Field Update
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
