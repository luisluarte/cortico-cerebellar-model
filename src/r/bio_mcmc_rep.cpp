
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio(int N_trials, mat X, vec ITI, vec reward,
                 mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                 vec params) {
    double p_alpha_pc   = params(0);
    double p_lambda_pc  = params(1);
    double p_beta_thal  = params(2);
    double p_kappa_cf   = params(3);
    double p_alpha_gran = params(4);
    double p_beta_gran  = params(5);
    double p_sigma2     = params(6);
    
    vec mu = zeros<vec>(32);
    vec Z = zeros<vec>(896);
    vec W_purk = zeros<vec>(896);
    vec D = zeros<vec>(362);
    
    mat mu_history = zeros<mat>(N_trials, 32);
    bool diverged = false;
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI(t) > 0) {
            Z.zeros();
            W_purk += randn<vec>(896) * sqrt(p_sigma2 * ITI(t));
        }
        
        vec I_t = X.row(t).t();
        vec I_hat = W_gen * mu; 
        vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI(t)) * mu + p_beta_thal * (W_thal * D)); 
        
        if(!mu.is_finite() || max(abs(mu)) > 1e4) {
            diverged = true;
            break;
        }
        
        vec G = W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = mean(abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        D = W_ach2 * (G % W_purk);
        
        mu_history.row(t) = mu.t();
    }
    
    if(diverged) {
        mat err(1,1); err.fill(NA_REAL);
        return err;
    }
    return mu_history;
}

double eval_empirical(mat bio_mu, vec y_choice, vec y_rt) {
    if(bio_mu.n_cols == 1 && !bio_mu.is_finite()) return -1e9;
    
    mat XX = bio_mu.t() * bio_mu;
    XX.diag() += 0.01;
    vec W_c;
    bool ok_c = solve(W_c, XX, bio_mu.t() * (y_choice - 0.5)); 
    if(!ok_c) return -1e9;
    vec logits = bio_mu * W_c;
    vec p = 1.0 / (1.0 + exp(-logits));
    p = clamp(p, 1e-7, 1.0 - 1e-7);
    double bce = accu(y_choice % log(p) + (1.0 - y_choice) % log(1.0 - p));
    
    vec W_rt;
    bool ok_rt = solve(W_rt, XX, bio_mu.t() * y_rt);
    if(!ok_rt) return -1e9;
    vec bio_rt = bio_mu * W_rt;
    vec res = y_rt - bio_rt;
    double sigma_rt = sqrt(mean(square(res))) + 1e-4;
    double rt_ll = accu(-0.5 * log(2.0 * M_PI * sigma_rt * sigma_rt) - square(res) / (2.0 * sigma_rt * sigma_rt));
    
    return bce + rt_ll;
}

double eval_distillation(mat bio_mu, mat rnn_mu) {
    if(bio_mu.n_cols == 1 && !bio_mu.is_finite()) return -1e9;
    mat XX = bio_mu.t() * bio_mu;
    XX.diag() += 0.01;
    mat W;
    bool ok = solve(W, XX, bio_mu.t() * rnn_mu);
    if(!ok) return -1e9;
    mat proj = bio_mu * W;
    mat res = rnn_mu - proj;
    double sigma2 = accu(square(res)) / res.n_elem + 1e-4;
    return accu(-0.5 * log(2.0 * M_PI * sigma2) - sum(square(res), 1) / (2.0 * sigma2));
}

// [[Rcpp::export]]
List run_bio_mcmc(int iters, int N_subjs, NumericVector subj_indices_r, 
                  NumericMatrix X_r, NumericVector ITI_r, NumericVector reward_r,
                  NumericVector human_choices_r, NumericVector y_rt_r, 
                  NumericMatrix rnn_mu_r, int N_total,
                  String target_mode, String prior_mode, NumericVector mu_EB_r,
                  NumericMatrix W_gen_r, NumericMatrix W_ach1_r, NumericMatrix W_ach2_r, NumericMatrix W_thal_r, NumericMatrix Pi_mat_r) {
                  
  vec subj_indices = as<vec>(subj_indices_r);
  mat X = as<mat>(X_r);
  vec ITI = as<vec>(ITI_r);
  vec reward = as<vec>(reward_r);
  vec human_choices = as<vec>(human_choices_r);
  vec y_rt = as<vec>(y_rt_r);
  mat rnn_mu = as<mat>(rnn_mu_r);
  vec mu_EB = as<vec>(mu_EB_r);
  
  mat W_gen = as<mat>(W_gen_r);
  mat W_ach1 = as<mat>(W_ach1_r);
  mat W_ach2 = as<mat>(W_ach2_r);
  mat W_thal = as<mat>(W_thal_r);
  mat Pi_mat = as<mat>(Pi_mat_r);
  
  vec mu_pop = zeros<vec>(7);
  if(prior_mode == "locked" || prior_mode == "anchored") {
    mu_pop = mu_EB;
  }
  
  vec sigma_pop = zeros<vec>(7); sigma_pop.fill(-2.0);
  mat Z_sub = zeros<mat>(N_subjs, 7);
  
  mat trace_mu = zeros<mat>(iters, 7);
  mat log_lik_trace = zeros<mat>(iters, N_total);
  
  double step_size = 0.05;
  
  double current_total_ll = 0;
  for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec params = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_sub.row(s).t());
      mat bio_mu = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), params);
      
      if(target_mode == "distillation") {
          current_total_ll += eval_distillation(bio_mu, rnn_mu.rows(idx));
      } else {
          current_total_ll += eval_empirical(bio_mu, human_choices.elem(idx), y_rt.elem(idx));
      }
  }
  
  for(int iter = 0; iter < iters; iter++) {
    
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
          mat bio_mu = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), params);
          if(target_mode == "distillation") new_total_ll += eval_distillation(bio_mu, rnn_mu.rows(idx));
          else new_total_ll += eval_empirical(bio_mu, human_choices.elem(idx), y_rt.elem(idx));
      }
      
      if(log(randu()) < (new_total_ll + prior_mu_new) - (current_total_ll + prior_mu_old)) {
          mu_pop = mu_pop_new;
          current_total_ll = new_total_ll;
      }
    }
    
    vec current_iter_ll = zeros<vec>(N_total);
    double step_total_ll = 0;
    
    for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec Z_old = Z_sub.row(s).t();
      vec Z_new = Z_old + randn<vec>(7) * step_size;
      
      vec params_old = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_old);
      mat bio_mu_old = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), params_old);
      double ll_old = target_mode == "distillation" ? eval_distillation(bio_mu_old, rnn_mu.rows(idx)) : eval_empirical(bio_mu_old, human_choices.elem(idx), y_rt.elem(idx));
      
      vec params_new = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_new);
      mat bio_mu_new = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), params_new);
      double ll_new = target_mode == "distillation" ? eval_distillation(bio_mu_new, rnn_mu.rows(idx)) : eval_empirical(bio_mu_new, human_choices.elem(idx), y_rt.elem(idx));
      
      if(log(randu()) < (ll_new + accu(-0.5*square(Z_new))) - (ll_old + accu(-0.5*square(Z_old)))) {
        Z_sub.row(s) = Z_new.t();
        current_iter_ll.elem(idx).fill(ll_new / idx.n_elem);
        step_total_ll += ll_new;
      } else {
        current_iter_ll.elem(idx).fill(ll_old / idx.n_elem);
        step_total_ll += ll_old;
      }
    }
    
    current_total_ll = step_total_ll;
    
    log_lik_trace.row(iter) = current_iter_ll.t();
    trace_mu.row(iter) = mu_pop.t();
  }
  
  return List::create(Named("log_lik_trace") = log_lik_trace);
}
