data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> D;
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y;
  matrix[N, D] trace;
}
parameters {
  real alpha_pop;
  real<lower=0> sigma_alpha;
  vector[K] alpha_raw;
  
  vector[D] W_pop;
  vector<lower=0>[D] sigma_W;
  matrix[D, K] W_raw;
}
transformed parameters {
  vector[K] alpha_subj = alpha_pop + sigma_alpha * alpha_raw;
  matrix[D, K] W_subj;
  for (k in 1:K) {
    W_subj[, k] = W_pop + sigma_W .* W_raw[, k];
  }
}
model {
  alpha_pop ~ normal(0, 3);
  sigma_alpha ~ normal(0, 1);
  alpha_raw ~ std_normal();
  
  W_pop ~ normal(0, 3);
  sigma_W ~ normal(0, 1);
  to_vector(W_raw) ~ std_normal();

  vector[N] logits;
  for (n in 1:N) {
    logits[n] = alpha_subj[subj_id[n]] + dot_product(row(trace, n), col(W_subj, subj_id[n]));
  }
  y ~ bernoulli_logit(logits);
}
generated quantities {
  vector[N] log_lik;
  vector[N] prob_pred;
  for (n in 1:N) {
    real l = alpha_subj[subj_id[n]] + dot_product(row(trace, n), col(W_subj, subj_id[n]));
    prob_pred[n] = inv_logit(l);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | l);
  }
}
