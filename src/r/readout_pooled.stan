
data {
  int<lower=1> N;
  int<lower=1> D; 
  array[N] int<lower=0,upper=1> y;
  matrix[N, D] trace;
}
parameters {
  vector[D] W_pop; // Pooled global readout weights (Capped at D parameters)
}
model {
  W_pop ~ normal(0, 3);
  vector[N] logits = trace * W_pop;
  y ~ bernoulli_logit(logits);
}
generated quantities {
  vector[N] log_lik;
  vector[N] logits = trace * W_pop;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | logits[n]);
  }
}
