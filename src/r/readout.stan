
data {
  int<lower=1> N; // total trials
  int<lower=1> K; // subjects
  int<lower=1> D; // Dimensions (32 for Bio, 8 for QLearn)
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y;
  matrix[N, D] trace;
}
parameters {
  vector[D] W_pop;
  vector<lower=0>[D] sigma_W;
  matrix[D, K] W_raw;
}
transformed parameters {
  matrix[D, K] W_subj;
  for (k in 1:K) {
    W_subj[, k] = W_pop + sigma_W .* W_raw[, k];
  }
}
model {
  W_pop ~ normal(0, 3);
  sigma_W ~ normal(0, 1);
  to_vector(W_raw) ~ std_normal();

  vector[N] logits;
  for (n in 1:N) {
    logits[n] = dot_product(row(trace, n), col(W_subj, subj_id[n]));
  }
  y ~ bernoulli_logit(logits);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | dot_product(row(trace, n), col(W_subj, subj_id[n])));
  }
}
