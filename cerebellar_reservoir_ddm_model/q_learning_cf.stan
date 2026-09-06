data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1,upper=S> subj;
  array[N] int<lower=1,upper=8> bd1;
  array[N] int<lower=1,upper=8> bd2;
  array[N] int<lower=1,upper=2> resp;
  array[N] real reward;
}

parameters {
  // Population level hypers
  real mu_aw_raw;  real<lower=0> sigma_aw_raw;
  real mu_al_raw;  real<lower=0> sigma_al_raw;
  real mu_beta_raw; real<lower=0> sigma_beta_raw;
  
  // Subject level non-centered parameters
  vector[S] aw_raw;
  vector[S] al_raw;
  vector[S] beta_raw;
}

transformed parameters {
  vector[S] alpha_win = inv_logit(mu_aw_raw + sigma_aw_raw * aw_raw);
  vector[S] alpha_loss = inv_logit(mu_al_raw + sigma_al_raw * al_raw);
  vector[S] beta = exp(mu_beta_raw + sigma_beta_raw * beta_raw);
}

model {
  // Priors
  mu_aw_raw ~ normal(0, 1.5); sigma_aw_raw ~ normal(0, 1);
  mu_al_raw ~ normal(0, 1.5); sigma_al_raw ~ normal(0, 1);
  mu_beta_raw ~ normal(0, 1.5); sigma_beta_raw ~ normal(0, 1);
  
  aw_raw ~ std_normal();
  al_raw ~ std_normal();
  beta_raw ~ std_normal();
  
  vector[8] Q;
  
  // Likelihood
  for (i in 1:N) {
    if (i == 1 || subj[i] != subj[i-1]) {
      Q = rep_vector(0.5, 8); // Reset Q-values for new subject
    }
    
    int b1 = bd1[i];
    int b2 = bd2[i];
    
    // Softmax Choice: P(resp=1) = inv_logit(beta * (Q[b1] - Q[b2]))
    // If resp=1, y=0. bernoulli_logit(0 | x) = inv_logit(-x)
    target += bernoulli_logit_lpmf(resp[i] - 1 | beta[subj[i]] * (Q[b2] - Q[b1]));
    
    // Q-learning update (Fictitious / Counterfactual included)
    int c = (resp[i] == 1) ? b1 : b2;
    int u = (resp[i] == 1) ? b2 : b1;
    real r = reward[i];
    real r_cf = 1.0 - r; // Assuming binary symmetric rewards for counterfactual
    
    // Update chosen option
    if (r > 0.5) {
      Q[c] += alpha_win[subj[i]] * (r - Q[c]);
    } else {
      Q[c] += alpha_loss[subj[i]] * (r - Q[c]);
    }
    
    // Update unchosen counterfactual option
    if (r_cf > 0.5) {
      Q[u] += alpha_win[subj[i]] * (r_cf - Q[u]);
    } else {
      Q[u] += alpha_loss[subj[i]] * (r_cf - Q[u]);
    }
  }
}
