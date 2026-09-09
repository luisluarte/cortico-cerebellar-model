
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio_subject(int subj_id, int N_trials, mat X, vec ITI, 
                         mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                         mat rand_noise,
                         vec params) {
  double p_alpha_pc   = params(0);
  double p_lambda_pc  = params(1);
  double p_beta_thal  = params(2);
  double p_kappa_cf   = params(3);
  double p_alpha_gran = params(4);
  double p_beta_gran  = params(5);
  double p_sigma2     = params(6);
  
  vec mu = zeros<vec>(32);
  vec Z  = zeros<vec>(896);
  vec W_purk = zeros<vec>(896);
  mat mu_history = zeros<mat>(N_trials, 32);
  
  for(int t = 0; t < N_trials; t++) {
    if(ITI(t) > 0) {
      Z.zeros();
      W_purk += rand_noise.row(t).t() * std::sqrt(p_sigma2 * ITI(t));
    }
    vec I_t = X.row(t).t();
    vec I_hat = W_gen * mu;
    vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
    vec G = W_ach1 * mu;
    Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
    double eps_mag = mean(abs(eps));
    W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
    vec D = W_ach2 * (G % W_purk);
    mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI(t)) * mu + p_beta_thal * (W_thal * D));
    
    if (!mu.is_finite() || max(abs(mu)) > 1e4) {
      mat err = zeros<mat>(N_trials, 32);
      err.fill(-999999.0);
      return err;
    }
    mu_history.row(t) = mu.t();
  }
  return mu_history;
}

struct LL_Result {
  double total_ll;
  vec pointwise_ll;
};

LL_Result eval_empirical_likelihood(mat mu_hist, vec y_choice, vec y_rt) {
  LL_Result res;
  int N = mu_hist.n_rows;
  
  if (mu_hist(0,0) == -999999.0) {
    res.pointwise_ll = zeros<vec>(N);
    res.pointwise_ll.fill(-999999.0);
    res.total_ll = -999999.0;
    return res;
  }
  
  mat XX = mu_hist.t() * mu_hist;
  XX.diag() += 0.01; 
  mat XX_inv = inv(XX);
  
  vec W_policy = XX_inv * (mu_hist.t() * y_choice);
  vec W_mu     = XX_inv * (mu_hist.t() * y_rt);
  
  vec bio_logits = mu_hist * W_policy;
  vec bio_mu     = mu_hist * W_mu;
  
  vec p = 1.0 / (1.0 + exp(-bio_logits));
  p = clamp(p, 1e-7, 1.0 - 1e-7);
  vec bce = y_choice % log(p) + (1.0 - y_choice) % log(1.0 - p);
  
  vec residuals = y_rt - bio_mu;
  double sigma_rt = std::sqrt(mean(square(residuals))) + 1e-4;
  
  vec rt_ll = zeros<vec>(N);
  for(int t = 0; t < N; t++) {
    rt_ll(t) = -0.5 * std::log(2.0 * M_PI * sigma_rt * sigma_rt) - (residuals(t) * residuals(t)) / (2.0 * sigma_rt * sigma_rt);
  }
  
  vec pointwise = bce + rt_ll;
  
  res.pointwise_ll = pointwise;
  res.total_ll = sum(pointwise);
  
  if (!res.pointwise_ll.is_finite()) {
    res.pointwise_ll.fill(-999999.0);
    res.total_ll = -999999.0;
  }
  
  return res;
}

// [[Rcpp::export]]
List run_empirical_mcmc_prior(int iters, int N_subjs, uvec subj_indices, 
                           mat X, vec ITI, vec y_choice, vec y_rt, mat Pi_mat,
                           mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, int N_total_trials) {
  
  ITI = ITI / max(ITI);
  vec mu_EB = {-0.491991, 0.195432, -6.754977, -3.496445, 0.075258, -0.982694, -2.459656};
  vec mu_pop = mu_EB; 
  vec sigma_pop = zeros<vec>(7); sigma_pop.fill(-2.0); // Slightly more relaxed variance than -3.0
  mat Z_sub = zeros<mat>(N_subjs, 7);
  mat global_rand_noise = randn<mat>(N_total_trials, 896);
  
  mat trace_mu = zeros<mat>(iters, 7);
  mat log_lik_trace = zeros<mat>(iters, N_total_trials);
  
  double step_size = 0.05; 
  
  for(int iter = 0; iter < iters; iter++) {
    
    // UPDATE MU_POP FREELY BUT WITH STRONG GAUSSIAN PRIOR AT MU_EB
    vec mu_pop_new = mu_pop + randn<vec>(7) * step_size;
    double ll_pop_old = 0;
    double ll_pop_new = 0;
    
    for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec z_s = Z_sub.row(s).t();
      
      vec params_old = inv_logit_scaled(mu_pop + exp(sigma_pop) % z_s);
      mat hist_old = simulate_bio_subject(s, idx.n_elem, X.rows(idx), ITI.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), global_rand_noise.rows(idx), params_old);
      ll_pop_old += eval_empirical_likelihood(hist_old, y_choice.elem(idx), y_rt.elem(idx)).total_ll;
      
      vec params_new = inv_logit_scaled(mu_pop_new + exp(sigma_pop) % z_s);
      mat hist_new = simulate_bio_subject(s, idx.n_elem, X.rows(idx), ITI.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), global_rand_noise.rows(idx), params_new);
      ll_pop_new += eval_empirical_likelihood(hist_new, y_choice.elem(idx), y_rt.elem(idx)).total_ll;
    }
    
    // EMPIRICAL BAYES PRIOR: N(mu_EB, sd=0.5)
    // -0.5 * ((x - mu)/sd)^2
    // sd=0.5 -> 1/sd^2 = 4
    double prior_mu_old = accu(-0.5 * square(mu_pop - mu_EB) * 4.0);
    double prior_mu_new = accu(-0.5 * square(mu_pop_new - mu_EB) * 4.0);
    
    if(log(randu()) < (ll_pop_new + prior_mu_new) - (ll_pop_old + prior_mu_old)) {
      mu_pop = mu_pop_new;
    }
    
    vec current_iter_ll = zeros<vec>(N_total_trials);
    for(int s = 0; s < N_subjs; s++) {
      uvec idx = find(subj_indices == s);
      vec Z_old = Z_sub.row(s).t();
      vec Z_new = Z_old + randn<vec>(7) * step_size;
      
      vec params_old = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_old);
      LL_Result res_old = eval_empirical_likelihood(
        simulate_bio_subject(s, idx.n_elem, X.rows(idx), ITI.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), global_rand_noise.rows(idx), params_old),
        y_choice.elem(idx), y_rt.elem(idx)
      );
      
      vec params_new = inv_logit_scaled(mu_pop + exp(sigma_pop) % Z_new);
      LL_Result res_new = eval_empirical_likelihood(
        simulate_bio_subject(s, idx.n_elem, X.rows(idx), ITI.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), global_rand_noise.rows(idx), params_new),
        y_choice.elem(idx), y_rt.elem(idx)
      );
      
      double prior_z_old = accu(-0.5 * square(Z_old));
      double prior_z_new = accu(-0.5 * square(Z_new));
      
      if(log(randu()) < (res_new.total_ll + prior_z_new) - (res_old.total_ll + prior_z_old)) {
        Z_sub.row(s) = Z_new.t();
        current_iter_ll.elem(idx) = res_new.pointwise_ll;
      } else {
        current_iter_ll.elem(idx) = res_old.pointwise_ll;
      }
    }
    trace_mu.row(iter) = mu_pop.t();
    log_lik_trace.row(iter) = current_iter_ll.t();
    if(iter % 10 == 0) Rcpp::Rcout << "Iteration " << iter << "\n";
  }
  
  return List::create(Named("log_lik_trace") = log_lik_trace, Named("trace_mu") = trace_mu);
}
