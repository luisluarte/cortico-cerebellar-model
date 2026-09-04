functions {
  real subject_lpmf(int subj_seq, array[] int subj_all, array[] int Bd1_all, array[] int Bd2_all, array[] int Resp_all, array[] real Reward_all, array[] real RT_all, array[] int start_idx, array[] int num_trials, real alpha_Q, real alpha_PC, real phys_decay, real g_s, real beta_v, real a_base, real ndt_base, real ndt_scale, vector kappa_vec, matrix inv_W_exp, int N_granules) {
    real lp = 0.0;
    int idx_offset = start_idx[subj_seq];
    int n_t = num_trials[subj_seq];
    
    vector[8] Q = rep_vector(0.0, 8);
    matrix[8, N_granules] W_PC_full = rep_matrix(0.0, 8, N_granules);
    vector[N_granules] Z = rep_vector(0.0, N_granules);
    
    for (t in 1:n_t) {
      int idx = idx_offset + t - 1;
      int c1 = Bd1_all[idx];
      int c2 = Bd2_all[idx];
      int ch = Resp_all[idx] == 1 ? c1 : c2;
      
      real Q_diff = Q[c1] - Q[c2];
      vector[N_granules] cortical_expansion = inv_W_exp * Q;
      
      Z = phys_decay * Z + kappa_vec .* cortical_expansion;
      vector[N_granules] Z_gated = Z .* tanh(g_s * sqrt(square(Z) + 1e-4));
      
      real Cb_c1 = dot_product(row(W_PC_full, c1), Z_gated);
      real Cb_c2 = dot_product(row(W_PC_full, c2), Z_gated);
      real Cb_diff = Cb_c1 - Cb_c2;
      
      // Topo 6: Logarithmic NDT Mapping, default smooth M
      real M = sqrt(square(Cb_diff - Q_diff) + 1e-4);
      real ndt_dyn = ndt_base + ndt_scale * log1p(M);
      
      real a_dyn = a_base;
      real v_t = beta_v * Q_diff;
      
      if (ndt_dyn > RT_all[idx] - 0.01) { ndt_dyn = RT_all[idx] - 0.01; }
      if (ndt_dyn < 0.01) { ndt_dyn = 0.01; }
      
      if (RT_all[idx] > ndt_dyn) {
        if (Resp_all[idx] == 1) { lp += wiener_lpdf(RT_all[idx] | a_dyn, ndt_dyn, 0.5, v_t); } 
        else { lp += wiener_lpdf(RT_all[idx] | a_dyn, ndt_dyn, 0.5, -v_t); }
      } else {
        lp += negative_infinity();
      }
      
      real pe = Reward_all[idx] - Q[ch];
      W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * pe * Z_gated)';
      Q[ch] = Q[ch] + alpha_Q * pe;
    }
    return lp;
  }
}
data {
  int<lower=1> N;
  int<lower=1> N_subj;
  array[N] int<lower=1> subj;
  array[N] int<lower=2, upper=8> Bd1;
  array[N] int<lower=2, upper=8> Bd2;
  array[N] int<lower=1, upper=2> Resp;
  array[N] real Reward;
  array[N] real RT;
}
transformed data {
  int N_granules = 50;
  matrix[N_granules, 8] inv_W_exp;
  for (i in 1:N_granules) {
    for (j in 1:8) { inv_W_exp[i, j] = normal_rng(0, 1 / sqrt(50.0)); }
  }
  array[N_subj] int subj_seq;
  array[N_subj] int start_idx;
  array[N_subj] int num_trials;
  int curr_s = 0;
  for (t in 1:N) {
    if (subj[t] != curr_s) {
      curr_s = subj[t];
      subj_seq[curr_s] = curr_s;
      start_idx[curr_s] = t;
      num_trials[curr_s] = 1;
    } else {
      num_trials[curr_s] += 1;
    }
  }
  
  array[N_subj] real min_RT_subj;
  for (s in 1:N_subj) { min_RT_subj[s] = 10.0; }
  for (t in 1:N) {
    if (RT[t] < min_RT_subj[subj[t]]) { min_RT_subj[subj[t]] = RT[t]; }
  }
}
parameters {
  real mu_alpha_Q; real<lower=0> sigma_alpha_Q; vector[N_subj] alpha_Q_raw;
  real mu_alpha_PC; real<lower=0> sigma_alpha_PC; vector[N_subj] alpha_PC_raw;
  real mu_phys; real<lower=0> sigma_phys; vector[N_subj] phys_raw;
  real mu_g_s; real<lower=0> sigma_g_s; vector[N_subj] g_s_raw;
  real mu_beta_v; real<lower=0> sigma_beta_v; vector[N_subj] beta_v_raw;
  real mu_a; real<lower=0> sigma_a; vector[N_subj] a_raw;
  real mu_ndt; real<lower=0> sigma_ndt; vector[N_subj] ndt_raw;
  real mu_scale; real<lower=0> sigma_scale; vector[N_subj] scale_raw;
  vector<lower=0, upper=1>[N_granules] kappa_vec;
}
transformed parameters {
  vector[N_subj] alpha_Q = inv_logit(mu_alpha_Q + sigma_alpha_Q * alpha_Q_raw);
  vector[N_subj] alpha_PC = inv_logit(mu_alpha_PC + sigma_alpha_PC * alpha_PC_raw);
  vector[N_subj] phys_decay = inv_logit(mu_phys + sigma_phys * phys_raw);
  vector[N_subj] g_s = exp(mu_g_s + sigma_g_s * g_s_raw);
  vector[N_subj] beta_v = exp(mu_beta_v + sigma_beta_v * beta_v_raw);
  vector[N_subj] a_base = exp(mu_a + sigma_a * a_raw);
  vector[N_subj] ndt_scale = exp(mu_scale + sigma_scale * scale_raw);
  
  vector[N_subj] ndt_base;
  for (s in 1:N_subj) {
    ndt_base[s] = min_RT_subj[s] * inv_logit(mu_ndt + sigma_ndt * ndt_raw[s]);
  }
}
model {
  mu_alpha_Q ~ normal(0, 1.5); sigma_alpha_Q ~ normal(0, 1); alpha_Q_raw ~ std_normal();
  mu_alpha_PC ~ normal(0, 1.5); sigma_alpha_PC ~ normal(0, 1); alpha_PC_raw ~ std_normal();
  mu_phys ~ normal(0, 1.5); sigma_phys ~ normal(0, 1); phys_raw ~ std_normal();
  mu_g_s ~ normal(0, 1); sigma_g_s ~ normal(0, 0.5); g_s_raw ~ std_normal();
  mu_beta_v ~ normal(0, 1); sigma_beta_v ~ normal(0, 0.5); beta_v_raw ~ std_normal();
  mu_a ~ normal(0.5, 0.5); sigma_a ~ normal(0, 0.5); a_raw ~ std_normal();
  mu_ndt ~ normal(-1, 1); sigma_ndt ~ normal(0, 1); ndt_raw ~ std_normal();
  mu_scale ~ normal(-1, 1); sigma_scale ~ normal(0, 1); scale_raw ~ std_normal();
  kappa_vec ~ beta(2, 2);
  
  for (s in 1:N_subj) {
    target += subject_lpmf(subj_seq[s] | subj, Bd1, Bd2, Resp, Reward, RT, start_idx, num_trials,
                           alpha_Q[s], alpha_PC[s], phys_decay[s], g_s[s], beta_v[s], a_base[s], ndt_base[s], ndt_scale[s],
                           kappa_vec, inv_W_exp, N_granules);
  }
}
generated quantities {
  real sum_log_lik = 0;
  for (s in 1:N_subj) {
    sum_log_lik += subject_lpmf(subj_seq[s] | subj, Bd1, Bd2, Resp, Reward, RT, start_idx, num_trials,
                           alpha_Q[s], alpha_PC[s], phys_decay[s], g_s[s], beta_v[s], a_base[s], ndt_base[s], ndt_scale[s],
                           kappa_vec, inv_W_exp, N_granules);
  }
}
