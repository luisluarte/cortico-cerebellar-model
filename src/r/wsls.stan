
data {
  int<lower=1> N;
  int<lower=1> K;
  array[N] int<lower=1,upper=K> subj_id;
  array[N] int<lower=0,upper=1> y;
  array[N] int<lower=0,upper=1> win;
  array[N] int<lower=0,upper=1> lag_c;
  real q3_theta_win_logit;
  real q3_theta_loss_logit;
}
parameters {
  real theta_win_pop_raw;
  real theta_loss_pop_raw;
  real<lower=0> sigma_win;
  real<lower=0> sigma_loss;
  vector[K] theta_win_subj_raw;
  vector[K] theta_loss_subj_raw;
}
transformed parameters {
  vector[K] theta_win_subj = inv_logit(theta_win_pop_raw + sigma_win * theta_win_subj_raw);
  vector[K] theta_loss_subj = inv_logit(theta_loss_pop_raw + sigma_loss * theta_loss_subj_raw);
}
model {
  theta_win_pop_raw ~ normal(q3_theta_win_logit, 0.5);
  theta_loss_pop_raw ~ normal(q3_theta_loss_logit, 0.5);
  sigma_win ~ normal(0, 0.5);
  sigma_loss ~ normal(0, 0.5);
  theta_win_subj_raw ~ std_normal();
  theta_loss_subj_raw ~ std_normal();

  for (n in 1:N) {
    real p_switch = win[n] == 1 ? theta_win_subj[subj_id[n]] : theta_loss_subj[subj_id[n]];
    real p_choice = lag_c[n] == 1 ? (1 - p_switch) : p_switch;
    y[n] ~ bernoulli(p_choice);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real p_switch = win[n] == 1 ? theta_win_subj[subj_id[n]] : theta_loss_subj[subj_id[n]];
    real p_choice = lag_c[n] == 1 ? (1 - p_switch) : p_switch;
    log_lik[n] = bernoulli_lpmf(y[n] | p_choice);
  }
}
