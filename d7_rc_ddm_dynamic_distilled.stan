functions {
  real subject_lpmf(array[] int subj_slice, int start, int end,
                    array[] int subj_ids_all,
                    array[] int Bd1_all, array[] int Bd2_all, array[] int Resp_all, array[] real Reward_all, array[] real RT_all,
                    vector soft_p_all, vector soft_mu_all, vector soft_sigma_all, vector soft_tau_all,
                    array[] int start_idx, array[] int num_trials,
                    real alpha_Q, real alpha_PC, real phys_decay, real g_s, real beta_v, real a_base, real a_scale, real ndt,
                    vector kappa_vec, matrix inv_W_exp, int N_granules, real lambda_policy) {
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
        vector[N_granules] cortical_expansion = inv_W_exp * Q;
        Z = phys_decay * Z + kappa_vec .* cortical_expansion;
        
        vector[N_granules] S_mask = tanh(g_s * sqrt(square(Z) + 1e-4));
        vector[N_granules] Z_gated = Z .* S_mask;
        
        real Cb_c1 = dot_product(row(W_PC_full, c1), Z_gated);
        real Cb_c2 = dot_product(row(W_PC_full, c2), Z_gated);
        real Cb_diff = Cb_c1 - Cb_c2;
        
        real M_align = sqrt(square(Cb_diff - Q_diff) + 1e-4);
        real a_dyn = a_base + a_scale * M_align;
        real v_t = beta_v * Q_diff;
        
        real log_prob_c1 = log_inv_logit(v_t * a_dyn);
        real log_prob_c2 = log1m_inv_logit(v_t * a_dyn);
        lp += lambda_policy * (soft_p_all[idx] * log_prob_c1 + (1 - soft_p_all[idx]) * log_prob_c2);
        
        real mu_hat_t = exp(soft_mu_all[idx] + square(soft_sigma_all[idx]) / 2.0) + soft_tau_all[idx];
        real v_abs = abs(v_t) + 1e-4;
        real expected_rt_wfpt = ndt + (a_dyn / (2.0 * v_abs)) * tanh((a_dyn * v_abs) / 2.0);
        
        real scale_mu = soft_sigma_all[idx] + 1e-4;
        lp += -0.5 * square((expected_rt_wfpt - mu_hat_t) / scale_mu);
        
        real pe = Reward_all[idx] - Q[ch];
        W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * pe * Z_gated)';
        Q[ch] = Q[ch] + alpha_Q * pe;
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
  
  vector[N] soft_p;
  vector[N] soft_mu;
  vector[N] soft_sigma;
  vector[N] soft_tau;
}
transformed data {
  int N_granules = 50;
  matrix[N_granules, 8] inv_W_exp;
  for (i in 1:N_granules) {
    for (j in 1:8) {
      inv_W_exp[i, j] = normal_rng(0, 1 / sqrt(50.0));
    }
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
  real<lower=0, upper=5> a_scale;
  real<lower=0, upper=min(RT)> ndt;
  vector<lower=0, upper=1>[N_granules] kappa_vec;
  real<lower=0.1, upper=10> lambda_policy;
}
model {
  alpha_Q ~ beta(2, 2);
  alpha_PC ~ beta(2, 2);
  phys_decay ~ beta(5, 2);
  g_s ~ normal(1, 1);
  beta_v ~ normal(0, 5);
  a_base ~ normal(1, 1);
  a_scale ~ normal(0, 1);
  ndt ~ uniform(0, min(RT));
  kappa_vec ~ beta(2, 2);
  lambda_policy ~ lognormal(0, 1);
  
  int grainsize = 1;
  target += reduce_sum(subject_lpmf, subj_seq, grainsize, 
                       subj, Bd1, Bd2, Resp, Reward, RT, soft_p, soft_mu, soft_sigma, soft_tau, start_idx, num_trials,
                       alpha_Q, alpha_PC, phys_decay, g_s, beta_v, a_base, a_scale, ndt,
                       kappa_vec, inv_W_exp, N_granules, lambda_policy);
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
      
      int c1 = Bd1[t];
      int c2 = Bd2[t];
      int ch = Resp[t] == 1 ? c1 : c2;
      
      real Q_diff = Q[c1] - Q[c2];
      vector[N_granules] cortical_expansion = inv_W_exp * Q;
      Z = phys_decay * Z + kappa_vec .* cortical_expansion;
      vector[N_granules] S_mask = tanh(g_s * sqrt(square(Z) + 1e-4));
      vector[N_granules] Z_gated = Z .* S_mask;
      real Cb_diff = dot_product(row(W_PC_full, c1), Z_gated) - dot_product(row(W_PC_full, c2), Z_gated);
      real M_align = sqrt(square(Cb_diff - Q_diff) + 1e-4);
      real a_dyn = a_base + a_scale * M_align;
      real v_t = beta_v * Q_diff;
      
      if (RT[t] > ndt) {
        if (Resp[t] == 1) {
          sum_log_lik += wiener_lpdf(RT[t] | a_dyn, ndt, 0.5, v_t);
        } else {
          sum_log_lik += wiener_lpdf(RT[t] | a_dyn, ndt, 0.5, -v_t);
        }
      }
      
      real pe = Reward[t] - Q[ch];
      W_PC_full[ch, :] = W_PC_full[ch, :] + (alpha_PC * pe * Z_gated)';
      Q[ch] = Q[ch] + alpha_Q * pe;
    }
  }
}
