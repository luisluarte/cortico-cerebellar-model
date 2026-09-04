functions {
  real subject_lpmf(array[] int subj_slice, int start, int end,
                    array[] int subj_ids_all,
                    array[] int Bd1_all, array[] int Bd2_all, array[] int Resp_all, array[] real Reward_all, array[] real RT_all,
                    array[] int start_idx, array[] int num_trials,
                    real alpha_Q, real alpha_PC, real phys_decay, real g_s, real beta_v, real a_base, real ndt_base, real ndt_scale,
                    vector kappa_vec, matrix inv_W_exp, int N_granules, int topo_id) {
    real lp = 0;
    for (s_idx in 1:size(subj_slice)) {
      int s = subj_slice[s_idx];
      int idx_offset = start_idx[s];
      int n_t = num_trials[s];
      
      vector[8] Q = rep_vector(0.0, 8);
      matrix[8, N_granules] W_PC_full = rep_matrix(0.0, 8, N_granules);
      vector[N_granules] Z = rep_vector(0.0, N_granules);
      
      for (t in 1:n_t) {
        int idx = idx_offset + t - 1;
        int c1 = Bd1_all[idx];
        int c2 = Bd2_all[idx];
        int ch = Resp_all[idx] == 1 ? c1 : c2;
        
        real Q_diff = Q[c1] - Q[c2];
        vector[N_granules] cortical_expansion;
        if (topo_id == 11) {
          cortical_expansion = inv_W_exp * softmax(Q);
        } else {
          cortical_expansion = inv_W_exp * Q;
        }
        
        Z = phys_decay * Z + kappa_vec .* cortical_expansion;
        
        vector[N_granules] Z_gated;
        if (topo_id == 12) {
          Z_gated = Z * tanh(g_s * sum(abs(Z)));
        } else {
          Z_gated = Z .* tanh(g_s * sqrt(square(Z) + 1e-4));
        }
        
        real Cb_c1 = dot_product(row(W_PC_full, c1), Z_gated);
        real Cb_c2 = dot_product(row(W_PC_full, c2), Z_gated);
        if (topo_id == 14) {
          Cb_c1 = tanh(Cb_c1);
          Cb_c2 = tanh(Cb_c2);
        }
        real Cb_diff = Cb_c1 - Cb_c2;
        
        real M;
        if (topo_id == 2) { M = abs(Cb_diff - Q_diff); }
        else if (topo_id == 3) { M = square(Cb_diff - Q_diff); }
        else if (topo_id == 4) { M = tanh(abs(Cb_diff - Q_diff)); }
        else if (topo_id == 7) { M = abs(Cb_c1 - Q[c1]) + abs(Cb_c2 - Q[c2]); }
        else if (topo_id == 8) { M = abs(Cb_diff); }
        else if (topo_id == 9) { M = (Q_diff > Cb_diff) ? (Q_diff - Cb_diff) : 0.0; }
        else if (topo_id == 10) { M = (Cb_diff > Q_diff) ? (Cb_diff - Q_diff) : 0.0; }
        else { M = sqrt(square(Cb_diff - Q_diff) + 1e-4); } // Default for 1, 5, 6, 11-20
        
        real ndt_dyn;
        if (topo_id == 5) { ndt_dyn = ndt_base * exp(ndt_scale * M); }
        else if (topo_id == 6) { ndt_dyn = ndt_base + ndt_scale * log1p(M); }
        else { ndt_dyn = ndt_base + ndt_scale * M; }
        
        real a_dyn = a_base;
        if (topo_id == 18) { a_dyn = a_base + ndt_scale * M; }
        
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
        
        if (topo_id == 17) { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * tanh(abs(pe)) * pe * Z_gated)'; }
        else if (topo_id == 19) { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * Reward_all[idx] * Z_gated)'; }
        else if (topo_id == 20) { W_PC_full[ch, :] = W_PC_full[ch, :] - (alpha_PC * pe * Z_gated)'; }
        else { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * pe * Z_gated)'; }
        
        if (topo_id == 13) { W_PC_full = 0.99 * W_PC_full; }
        
        if (topo_id == 15) { Q[ch] = Q[ch] + alpha_Q * (1.0 - tanh(M)) * pe; }
        else if (topo_id == 16) { Q[ch] = Q[ch] + alpha_Q * (1.0 + M) * pe; }
        else { Q[ch] = Q[ch] + alpha_Q * pe; }
      }
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
  int<lower=1, upper=20> topo_id;
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
}
parameters {
  real<lower=0, upper=1> alpha_Q;
  real<lower=0, upper=1> alpha_PC;
  real<lower=0, upper=1> phys_decay;
  real<lower=0, upper=5> g_s;
  real<lower=0, upper=20> beta_v;
  real<lower=0, upper=5> a_base;
  real<lower=0, upper=min(RT)> ndt_base;
  real<lower=0, upper=2> ndt_scale;
  vector<lower=0, upper=1>[N_granules] kappa_vec;
}
model {
  alpha_Q ~ beta(2, 2);
  alpha_PC ~ beta(2, 2);
  phys_decay ~ beta(5, 2);
  g_s ~ normal(1, 1);
  beta_v ~ normal(0, 5);
  a_base ~ normal(1, 1);
  ndt_base ~ uniform(0, min(RT));
  ndt_scale ~ normal(0, 0.5);
  kappa_vec ~ beta(2, 2);
  
  target += reduce_sum(subject_lpmf, subj_seq, 1, 
                       subj, Bd1, Bd2, Resp, Reward, RT, start_idx, num_trials,
                       alpha_Q, alpha_PC, phys_decay, g_s, beta_v, a_base, ndt_base, ndt_scale,
                       kappa_vec, inv_W_exp, N_granules, topo_id);
}
generated quantities {
  real sum_log_lik = 0;
  {
    vector[8] Q;
    matrix[8, N_granules] W_PC_full; 
    vector[N_granules] Z;
    int current_subj = -1;
    for (t in 1:N) {
      if (subj[t] != current_subj) {
        Q = rep_vector(0.0, 8);
        Z = rep_vector(0.0, N_granules);
        W_PC_full = rep_matrix(0.0, 8, N_granules);
        current_subj = subj[t];
      }
      int c1 = Bd1[t]; int c2 = Bd2[t]; int ch = Resp[t] == 1 ? c1 : c2;
      real Q_diff = Q[c1] - Q[c2];
      vector[N_granules] cortical_expansion;
      if (topo_id == 11) { cortical_expansion = inv_W_exp * softmax(Q); } 
      else { cortical_expansion = inv_W_exp * Q; }
      Z = phys_decay * Z + kappa_vec .* cortical_expansion;
      vector[N_granules] Z_gated;
      if (topo_id == 12) { Z_gated = Z * tanh(g_s * sum(abs(Z))); } 
      else { Z_gated = Z .* tanh(g_s * sqrt(square(Z) + 1e-4)); }
      real Cb_c1 = dot_product(row(W_PC_full, c1), Z_gated);
      real Cb_c2 = dot_product(row(W_PC_full, c2), Z_gated);
      if (topo_id == 14) { Cb_c1 = tanh(Cb_c1); Cb_c2 = tanh(Cb_c2); }
      real Cb_diff = Cb_c1 - Cb_c2;
      
      real M;
      if (topo_id == 2) { M = abs(Cb_diff - Q_diff); }
      else if (topo_id == 3) { M = square(Cb_diff - Q_diff); }
      else if (topo_id == 4) { M = tanh(abs(Cb_diff - Q_diff)); }
      else if (topo_id == 7) { M = abs(Cb_c1 - Q[c1]) + abs(Cb_c2 - Q[c2]); }
      else if (topo_id == 8) { M = abs(Cb_diff); }
      else if (topo_id == 9) { M = (Q_diff > Cb_diff) ? (Q_diff - Cb_diff) : 0.0; }
      else if (topo_id == 10) { M = (Cb_diff > Q_diff) ? (Cb_diff - Q_diff) : 0.0; }
      else { M = sqrt(square(Cb_diff - Q_diff) + 1e-4); }
      
      real ndt_dyn;
      if (topo_id == 5) { ndt_dyn = ndt_base * exp(ndt_scale * M); }
      else if (topo_id == 6) { ndt_dyn = ndt_base + ndt_scale * log1p(M); }
      else { ndt_dyn = ndt_base + ndt_scale * M; }
      
      real a_dyn = a_base;
      if (topo_id == 18) { a_dyn = a_base + ndt_scale * M; }
      real v_t = beta_v * Q_diff;
      if (ndt_dyn > RT[t] - 0.01) { ndt_dyn = RT[t] - 0.01; }
      if (ndt_dyn < 0.01) { ndt_dyn = 0.01; }
      
      if (RT[t] > ndt_dyn) {
        if (Resp[t] == 1) { sum_log_lik += wiener_lpdf(RT[t] | a_dyn, ndt_dyn, 0.5, v_t); } 
        else { sum_log_lik += wiener_lpdf(RT[t] | a_dyn, ndt_dyn, 0.5, -v_t); }
      }
      
      real pe = Reward[t] - Q[ch];
      if (topo_id == 17) { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * tanh(abs(pe)) * pe * Z_gated)'; }
      else if (topo_id == 19) { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * Reward[t] * Z_gated)'; }
      else if (topo_id == 20) { W_PC_full[ch, :] = W_PC_full[ch, :] - (alpha_PC * pe * Z_gated)'; }
      else { W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * pe * Z_gated)'; }
      if (topo_id == 13) { W_PC_full = 0.99 * W_PC_full; }
      if (topo_id == 15) { Q[ch] = Q[ch] + alpha_Q * (1.0 - tanh(M)) * pe; }
      else if (topo_id == 16) { Q[ch] = Q[ch] + alpha_Q * (1.0 + M) * pe; }
      else { Q[ch] = Q[ch] + alpha_Q * pe; }
    }
  }
}
