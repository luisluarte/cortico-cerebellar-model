
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio_subject(int N_trials, mat X, vec ITI, vec reward,
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

// [[Rcpp::export]]
List run_bio_mcmc_chain_cov(int iters, NumericMatrix X_r, NumericVector ITI_r, NumericVector reward_r,
                        NumericVector human_choices_r, NumericVector y_rt_r, NumericMatrix rnn_mu_r,
                        String target_mode, String prior_mode, NumericVector mu_EB_r,
                        NumericMatrix W_gen_r, NumericMatrix W_ach1_r, NumericMatrix W_ach2_r, NumericMatrix W_thal_r, NumericMatrix Pi_mat_r) {
                  
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
  
  int N_trials = X.n_rows;
  
  vec mu_pop = zeros<vec>(7);
  if(prior_mode == "locked" || prior_mode == "anchored") mu_pop = mu_EB;
  
  vec sigma_pop = zeros<vec>(7);
  if(prior_mode == "flat") {
      sigma_pop.fill(2.0);
  } else if(prior_mode == "locked") {
      sigma_pop.fill(-3.0); 
  } else if(prior_mode == "anchored") {
      sigma_pop.fill(0.0);  
  }
  
  mat trace_z = zeros<mat>(iters, 7);
  mat trace_ll = zeros<mat>(iters, N_trials);
  
  double step_size = 0.05;
  
  // Empirical Covariance Objects
  mat Sigma = eye<mat>(7,7);
  mat L = eye<mat>(7,7);
  double gamma = 2.38 / sqrt(7.0);
  
  vec Z_curr = randn<vec>(7) * 0.01; 
  vec params_curr = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_curr);
  mat bio_mu_curr = simulate_bio_subject(N_trials, X, ITI, reward, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, params_curr);
  double ll_curr = target_mode == "distillation" ? eval_distillation(bio_mu_curr, rnn_mu) : eval_empirical(bio_mu_curr, human_choices, y_rt);
  
  for(int iter = 0; iter < iters; iter++) {
    
    // Periodically update the covariance matrix after burn-in starts (500 steps)
    if (iter >= 500 && iter % 50 == 0) {
        mat trace_sub = trace_z.rows(250, iter - 1); 
        Sigma = cov(trace_sub);
        Sigma.diag() += 1e-5; // Add nugget for positive definiteness
        bool ok = chol(L, Sigma, "lower");
        if(!ok) { L = eye<mat>(7,7) * step_size; }
    }
    
    vec step;
    if (iter < 500) {
        // Phase 1: Spherical Scalar Random Walk
        step = randn<vec>(7) * step_size;
    } else {
        // Phase 2: Stretched Empirical Covariance Random Walk (Haario 2001)
        if (randu() < 0.95) {
            step = gamma * (L * randn<vec>(7));
        } else {
            step = randn<vec>(7) * 0.01; // 5% small isotropic jumps to guarantee ergodicity
        }
    }
    
    vec Z_new = Z_curr + step;
    vec params_new = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_new);
    mat bio_mu_new = simulate_bio_subject(N_trials, X, ITI, reward, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, params_new);
    double ll_new = target_mode == "distillation" ? eval_distillation(bio_mu_new, rnn_mu) : eval_empirical(bio_mu_new, human_choices, y_rt);
    
    double prior_old = accu(-0.5*square(Z_curr));
    double prior_new = accu(-0.5*square(Z_new));
    
    double log_alpha = (ll_new + prior_new) - (ll_curr + prior_old);
    if(std::isnan(log_alpha)) log_alpha = -1e9;
    
    double accept_prob = exp(log_alpha);
    if(accept_prob > 1.0) accept_prob = 1.0;
    
    if(log(randu()) < log_alpha) {
        Z_curr = Z_new;
        ll_curr = ll_new;
    }
    
    // Adaptive step size only during Phase 1
    if(iter < 500) {
        double eta = 1.0 / sqrt(iter + 1.0);
        step_size = exp(log(step_size) + eta * (accept_prob - 0.234));
        if(step_size > 5.0) step_size = 5.0;
        if(step_size < 1e-4) step_size = 1e-4;
    }
    
    trace_z.row(iter) = Z_curr.t();
    trace_ll.row(iter).fill(ll_curr / N_trials); 
  }
  
  return List::create(Named("trace_z") = trace_z, Named("trace_ll") = trace_ll);
}
