
data {
  int<lower=1> N;
  array[N] int<lower=0,upper=1> y;
  array[N] int<lower=0,upper=1> win;
  real q3_theta_win_logit;
  real q3_theta_loss_logit;
}
parameters {
  real theta_win_pop_raw;
  real theta_loss_pop_raw;
}
transformed parameters {
  real theta_win_pop = inv_logit(theta_win_pop_raw);
  real theta_loss_pop = inv_logit(theta_loss_pop_raw);
}
model {
  theta_win_pop_raw ~ normal(q3_theta_win_logit, 0.5);
  theta_loss_pop_raw ~ normal(q3_theta_loss_logit, 0.5);

  for (n in 1:N) {
    real p_choice = win[n] == 1 ? theta_win_pop : theta_loss_pop;
    y[n] ~ bernoulli(p_choice);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real p_choice = win[n] == 1 ? theta_win_pop : theta_loss_pop;
    log_lik[n] = bernoulli_lpmf(y[n] | p_choice);
  }
}
