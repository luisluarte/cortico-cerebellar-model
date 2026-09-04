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
  alpha_pos ~ beta(2, 2);
  alpha_neg ~ beta(2, 2);
  beta_v ~ normal(0, 5);
  a ~ normal(1, 1);
  ndt ~ uniform(0, min(RT));
  
  matrix[8, n_dim] Theta;
  int current_subj = -1;
  
  for (t in 1:N) {
    if (subj[t] != current_subj) {
      Theta = rep_matrix(0.0, 8, n_dim);
      current_subj = subj[t];
    }
    
    int c1 = Bd1[t];
    int c2 = Bd2[t];
    int ch = Resp[t] == 1 ? c1 : c2;
    int unch = Resp[t] == 1 ? c2 : c1;
    
    // Multi-hot input vector (only non-zero at c1 and c2)
    // We can just add the columns of W_exp directly instead of building the vector X
    vector[n_dim] phi_X = tanh(W_exp[:, c1] + W_exp[:, c2]);
    
    // Compression to value space
    real V_ch = dot_product(Theta[ch]', phi_X);
    real V_unch = dot_product(Theta[unch]', phi_X);
    real v_t = beta_v * (V_ch - V_unch);
    
    if (Resp[t] == 1) {
      RT[t] ~ wiener(a, ndt, 0.5, v_t);
    } else {
      RT[t] ~ wiener(a, ndt, 0.5, -v_t);
    }
    
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      Theta[ch] = Theta[ch] + (alpha_pos * E_t * phi_X)';
    } else {
      Theta[ch] = Theta[ch] + (alpha_neg * E_t * phi_X)';
    }
  }
}
generated quantities {
  vector[N] log_lik;
  matrix[8, n_dim] Theta_gen;
  int current_subj_gen = -1;
  
  for (t in 1:N) {
    if (subj[t] != current_subj_gen) {
      Theta_gen = rep_matrix(0.0, 8, n_dim);
      current_subj_gen = subj[t];
    }
    
    int c1 = Bd1[t];
    int c2 = Bd2[t];
    int ch = Resp[t] == 1 ? c1 : c2;
    int unch = Resp[t] == 1 ? c2 : c1;
    
    vector[n_dim] phi_X = tanh(W_exp[:, c1] + W_exp[:, c2]);
    
    real V1 = dot_product(Theta_gen[c1]', phi_X);
    real V2 = dot_product(Theta_gen[c2]', phi_X);
    real v_t = beta_v * (V1 - V2);
    
    if (Resp[t] == 1) {
      log_lik[t] = wiener_lpdf(RT[t] | a, ndt, 0.5, v_t);
    } else {
      log_lik[t] = wiener_lpdf(RT[t] | a, ndt, 0.5, -v_t);
    }
    
    real V_ch = dot_product(Theta_gen[ch]', phi_X);
    real E_t = Reward[t] - V_ch;
    if (E_t > 0) {
      Theta_gen[ch] = Theta_gen[ch] + (alpha_pos * E_t * phi_X)';
    } else {
      Theta_gen[ch] = Theta_gen[ch] + (alpha_neg * E_t * phi_X)';
    }
  }
}
