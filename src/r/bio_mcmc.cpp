
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio(int N_trials, mat X, vec ITI, vec reward, vec params) {
  double p_alpha_pc   = params(0);
  double p_lambda_pc  = params(1);
  double p_beta_thal  = params(2);
  double p_kappa_cf   = params(3);
  double p_alpha_gran = params(4);
  double p_beta_gran  = params(5);
  double p_sigma2     = params(6);
  
  vec mu = zeros<vec>(32);
  vec Z = zeros<vec>(32);
  mat W_purk = zeros<mat>(32, 32);
  mat mu_history = zeros<mat>(N_trials, 32);
  
  for(int t = 0; t < N_trials; t++) {
    vec stim = zeros<vec>(32);
    stim(0) = X(t, 0);
    stim(1) = X(t, 1);
    
    Z = p_alpha_gran * Z + p_beta_gran * stim;
    vec purk = W_purk * Z;
    mu = p_alpha_pc * mu + p_beta_thal * purk;
    
    if(!mu.is_finite() || max(abs(mu)) > 1e4) {
      mu.fill(1e4);
      mu_history.row(t) = mu.t();
      continue;
    }
    
    mu_history.row(t) = mu.t();
    
    double r = reward[t];
    double cf_fire = 1.0 / (1.0 + exp(-r * p_kappa_cf));
    W_purk += p_lambda_pc * cf_fire * (Z * mu.t());
    
    double dt = ITI[t];
    Z *= exp(-dt * 0.1);
    mu *= exp(-dt * p_sigma2);
  }
  return mu_history;
}

double eval_subject_ll(int s, uvec idx, mat X, vec ITI, vec reward, 
                       vec human_choices, vec rnn_logits, mat rnn_mu, 
                       vec params_scaled, String target_mode) {
    
    mat bio_mu = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), params_scaled);
    
    if(target_mode == "distillation") {
        mat XX = bio_mu.t() * bio_mu;
        XX.diag() += 0.01;
        mat W;
        bool ok = solve(W, XX, bio_mu.t() * rnn_mu.rows(idx));
        if(!ok) W = zeros<mat>(32, rnn_mu.n_cols);
        mat proj = bio_mu * W;
        
        mat res = rnn_mu.rows(idx) - proj;
        double sigma2 = accu(square(res)) / res.n_elem + 1e-4;
        vec point_ll = -0.5 * log(2.0 * M_PI * sigma2) - sum(square(res), 1) / (2.0 * sigma2);
        return accu(point_ll);
        
    } else {
        mat XX = bio_mu.t() * bio_mu;
        XX.diag() += 0.01;
        vec W;
        bool ok = solve(W, XX, bio_mu.t() * rnn_logits.elem(idx));
        if(!ok) W = zeros<vec>(32);
        
        vec logits = bio_mu * W;
        vec p = 1.0 / (1.0 + exp(-logits));
        p = clamp(p, 1e-7, 1.0 - 1e-7);
        vec point_ll = human_choices.elem(idx) % log(p) + (1.0 - human_choices.elem(idx)) % log(1.0 - p);
        return accu(point_ll);
    }
}

vec get_subject_point_ll(int s, uvec idx, mat X, vec ITI, vec reward, 
                         vec human_choices, vec rnn_logits, mat rnn_mu, 
                         vec params_scaled, String target_mode) {
    
    mat bio_mu = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), params_scaled);
    
    if(target_mode == "distillation") {
        mat XX = bio_mu.t() * bio_mu;
        XX.diag() += 0.01;
        mat W;
        bool ok = solve(W, XX, bio_mu.t() * rnn_mu.rows(idx));
        if(!ok) W = zeros<mat>(32, rnn_mu.n_cols);
        mat proj = bio_mu * W;
        
        mat res = rnn_mu.rows(idx) - proj;
        double sigma2 = accu(square(res)) / res.n_elem + 1e-4;
        return -0.5 * log(2.0 * M_PI * sigma2) - sum(square(res), 1) / (2.0 * sigma2);
        
    } else {
        mat XX = bio_mu.t() * bio_mu;
        XX.diag() += 0.01;
        vec W;
        bool ok = solve(W, XX, bio_mu.t() * rnn_logits.elem(idx));
        if(!ok) W = zeros<vec>(32);
        
        vec logits = bio_mu * W;
        vec p = 1.0 / (1.0 + exp(-logits));
        p = clamp(p, 1e-7, 1.0 - 1e-7);
        return human_choices.elem(idx) % log(p) + (1.0 - human_choices.elem(idx)) % log(1.0 - p);
    }
}

// [[Rcpp::export]]
List run_bio_mcmc(int iters, int N_subjs, NumericVector subj_indices_r, 
                  NumericMatrix X_r, NumericVector ITI_r, NumericVector reward_r,
                  NumericVector human_choices_r, NumericVector rnn_logits_r,
                  NumericMatrix rnn_mu_r, int N_total,
                  String target_mode, String prior_mode, NumericVector mu_EB_r) {
                  
  vec subj_indices = as<vec>(subj_indices_r);
  mat X = as<mat>(X_r);
  vec ITI = as<vec>(ITI_r);
  vec reward = as<vec>(reward_r);
  vec human_choices = as<vec>(human_choices_r);
  vec rnn_logits = as<vec>(rnn_logits_r);
  mat rnn_mu = as<mat>(rnn_mu_r);
  vec mu_EB = as<vec>(mu_EB_r);
  
  vec mu_pop = zeros<vec>(7);
  if(prior_mode == "locked" || prior_mode == "anchored") {
    mu_pop = mu_EB;
  }
  
  vec sigma_pop = zeros<vec>(7); sigma_pop.fill(-2.0);
  mat Z_sub = zeros<mat>(N_subjs, 7);
  
  mat trace_mu = zeros<mat>(iters, 7);
  mat log_lik_trace = zeros<mat>(iters, N_total);
  
  double step_size = 0.05;
  
  // Calculate initial total likelihood
  double current_total_ll = 0;
  for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec params = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_sub.row(s).t());
      current_total_ll += eval_subject_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params, target_mode);
  }
  
  for(int iter = 0; iter < iters; iter++) {
    
    // 1. Update mu_pop (Global update)
    if(prior_mode != "locked") {
      vec mu_pop_new = mu_pop + randn<vec>(7) * 0.02;
      double prior_mu_old = 0; double prior_mu_new = 0;
      if(prior_mode == "anchored") {
        prior_mu_old = accu(-0.5 * square(mu_pop - mu_EB) * 4.0);
        prior_mu_new = accu(-0.5 * square(mu_pop_new - mu_EB) * 4.0);
      } else {
        prior_mu_old = accu(-0.5 * square(mu_pop) / 9.0);
        prior_mu_new = accu(-0.5 * square(mu_pop_new) / 9.0);
      }
      
      double new_total_ll = 0;
      for(int s = 0; s < N_subjs; s++) {
          uvec idx = find(subj_indices == s);
          vec params = inv_logit_scaled(mu_pop_new + exp(sigma_pop) % Z_sub.row(s).t());
          new_total_ll += eval_subject_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params, target_mode);
      }
      
      if(log(randu()) < (new_total_ll + prior_mu_new) - (current_total_ll + prior_mu_old)) {
          mu_pop = mu_pop_new;
          current_total_ll = new_total_ll;
      }
    }
    
    // 2. Update Z_sub (Local updates)
    vec current_iter_ll = zeros<vec>(N_total);
    double step_total_ll = 0;
    
    for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec Z_old = Z_sub.row(s).t();
      vec Z_new = Z_old + randn<vec>(7) * step_size;
      
      vec params_old = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_old);
      double ll_old = eval_subject_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params_old, target_mode);
      
      vec params_new = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_new);
      double ll_new = eval_subject_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params_new, target_mode);
      
      if(log(randu()) < (ll_new + accu(-0.5*square(Z_new))) - (ll_old + accu(-0.5*square(Z_old)))) {
        Z_sub.row(s) = Z_new.t();
        current_iter_ll.elem(idx) = get_subject_point_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params_new, target_mode);
        step_total_ll += ll_new;
      } else {
        current_iter_ll.elem(idx) = get_subject_point_ll(s, idx, X, ITI, reward, human_choices, rnn_logits, rnn_mu, params_old, target_mode);
        step_total_ll += ll_old;
      }
    }
    
    current_total_ll = step_total_ll;
    
    log_lik_trace.row(iter) = current_iter_ll.t();
    trace_mu.row(iter) = mu_pop.t();
  }
  
  return List::create(Named("log_lik_trace") = log_lik_trace, Named("trace_mu") = trace_mu);
}
