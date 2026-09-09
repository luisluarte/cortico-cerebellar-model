
library(Rcpp)
library(RcppArmadillo)

targets <- readRDS("distillation_targets_v2.rds")
unique_subjs <- unique(targets$subjs)
test_subjs <- unique_subjs[1:50]
idx <- which(targets$subjs %in% test_subjs)

data_list <- list(
  N_total = length(idx),
  subj_indices = as.numeric(factor(targets$subjs[idx])) - 1,
  X = as.matrix(targets$X[idx, ]),
  ITI = as.numeric(targets$ITI[idx]),
  y_choice = pmax(pmin(plogis(as.numeric(targets$rnn_logits[idx])), 1-1e-7), 1e-7),
  y_rt = matrix(rep(as.numeric(targets$rnn_mu[idx]), 32), ncol=32),
  y_sigma = as.numeric(targets$rnn_sigma[idx])
)

set.seed(42)
W_gen <- matrix(rnorm(6 * 32, 0, 1/sqrt(32)), nrow=6, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)

Pi_mat <- matrix(0, nrow=data_list$N_total, ncol=6)
for(s in 0:(length(test_subjs)-1)) {
  s_idx <- which(data_list$subj_indices == s)
  subj_var <- apply(data_list$X[s_idx, ], 2, var)
  raw_prec <- 1.0 / (subj_var + 1e-6)
  Pi_mat[s_idx, ] <- matrix(rep(raw_prec / mean(raw_prec), length(s_idx)), nrow=length(s_idx), byrow=TRUE)
}

set.seed(42)
global_rand_noise <- matrix(rnorm(data_list$N_total * 896), nrow=data_list$N_total)

mu_EB <- c(-0.491991, 0.195432, -6.754977, -3.496445, 0.075258, -0.982694, -2.459656)

cpp_code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio_subject(int subj_id, int N_trials, mat X, vec ITI, 
                         mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                         mat rand_noise, vec params) {
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
      mat err = zeros<mat>(N_trials, 32); err.fill(-999999.0); return err;
    }
    mu_history.row(t) = mu.t();
  }
  return mu_history;
}

// [[Rcpp::export]]
List eval_bio_predictions(int N_subjs, uvec subj_indices, mat X, vec ITI, 
                          mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                          mat rand_noise, vec mu_EB, vec y_choice, mat y_rt) {
  ITI = ITI / max(ITI);
  vec params = inv_logit_scaled(mu_EB);
  
  int N_total = X.n_rows;
  vec bio_p = zeros<vec>(N_total);
  mat bio_y_rt = zeros<mat>(N_total, 32);
  
  for(int s = 0; s < N_subjs; s++) {
    uvec idx = find(subj_indices == s);
    mat mu_hist = simulate_bio_subject(s, idx.n_elem, X.rows(idx), ITI.elem(idx), 
                                       W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), 
                                       rand_noise.rows(idx), params);
    
    mat XX = mu_hist.t() * mu_hist;
    XX.diag() += 0.01; 
    mat XX_inv = inv(XX);
    vec W_policy = XX_inv * (mu_hist.t() * y_choice.elem(idx));
    mat W_mu     = XX_inv * (mu_hist.t() * y_rt.rows(idx));
    vec bio_logits = mu_hist * W_policy;
    vec p = 1.0 / (1.0 + exp(-bio_logits));
    p = clamp(p, 1e-7, 1.0 - 1e-7);
    
    bio_p.elem(idx) = p;
    bio_y_rt.rows(idx) = mu_hist * W_mu;
  }
  return List::create(Named("y_hat_choice") = bio_p, Named("y_hat_rt") = bio_y_rt);
}
'

sourceCpp(code = cpp_code)

res <- eval_bio_predictions(50, data_list$subj_indices, data_list$X, data_list$ITI, 
                            W_gen, W_ach1, W_ach2, W_thal, Pi_mat,
                            global_rand_noise, mu_EB, data_list$y_choice, data_list$y_rt)

y_hat_choice <- as.numeric(res$y_hat_choice)
y_hat_rt <- as.matrix(res$y_hat_rt)
true_labels <- targets$labels[idx]

calc_pr_auc <- function(probs, labels) {
  ord <- order(probs, decreasing=TRUE)
  probs <- probs[ord]
  labels <- labels[ord]
  tp <- cumsum(labels == 1)
  fp <- cumsum(labels == 0)
  precision <- tp / (tp + fp)
  recall <- tp / sum(labels == 1)
  auc <- sum(diff(recall) * (precision[-1] + precision[-length(precision)]) / 2)
  return(auc)
}

pr_auc <- calc_pr_auc(y_hat_choice, true_labels)
cat("Biological Model PR-AUC (vs Empirical Choice): ", pr_auc, "\n")

rt_rmse <- sqrt(mean((y_hat_rt - data_list$y_rt)^2))
cat("Biological Model RT-RMSE (vs RNN representation): ", rt_rmse, "\n")
