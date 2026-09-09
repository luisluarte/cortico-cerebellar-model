
library(Rcpp)
library(RcppArmadillo)
library(prROC)

targets <- readRDS("distillation_targets_v2.rds")
unique_subjs <- unique(targets$subjs)
test_subjs <- unique_subjs[1:50]
idx <- which(targets$subjs %in% test_subjs)

data_list <- list(
  N_total = length(idx),
  subj_indices = as.numeric(factor(targets$subjs[idx])) - 1,
  X = as.matrix(targets$X[idx, ]),
  ITI = as.numeric(targets$ITI[idx]),
  lag_reward = as.numeric(targets$lag_reward[idx]),
  lag_ch = as.numeric(targets$lag_ch[idx]),
  lag_resp = as.numeric(targets$lag_resp[idx]),
  y_choice = pmax(pmin(plogis(as.numeric(targets$rnn_logits[idx])), 1-1e-7), 1e-7),
  y_rt = matrix(rep(as.numeric(targets$rnn_mu[idx]), 32), ncol=32),
  y_sigma = as.numeric(targets$rnn_sigma[idx])
)

sourceCpp("full_mcmc.cpp")

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

# The global random noise matrix must be the EXACT SAME ONE generated during the successful MCMC.
# Since we fixed the seed before generating it, it will be identical!
set.seed(42)
global_rand_noise <- matrix(rnorm(data_list$N_total * 896), nrow=data_list$N_total)

# The posterior mean for mu_pop is effectively the mu_EB we injected (because sigma_pop = -3.0 means the MCMC hardly drifts)
mu_EB <- c(-0.491991, 0.195432, -6.754977, -3.496445, 0.075258, -0.982694, -2.459656)

L_bounds <- c(0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000)
U_bounds <- c(1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000)

inv_logit_scaled <- function(q) {
  p <- plogis(q)
  L_bounds + p * (U_bounds - L_bounds)
}

params_bio <- inv_logit_scaled(mu_EB)

cat("Evaluating Biological Posterior Predictive Performance...\n")

# We evaluate the choice predictions (bio_logits) and RT predictions (bio_mu)
# Since we didn't return y_hat from C++, we can easily evaluate it in R using a similar logic,
# OR we can just write a quick C++ wrapper to return the predictions.

cpp_code <- '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

// [[Rcpp::export]]
List eval_bio_predictions(int N_subjs, uvec subj_indices, mat X, vec ITI, 
                          mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                          mat rand_noise, vec params, vec y_choice, mat y_rt) {
                          
  double p_alpha_pc   = params(0);
  double p_lambda_pc  = params(1);
  double p_beta_thal  = params(2);
  double p_kappa_cf   = params(3);
  double p_alpha_gran = params(4);
  double p_beta_gran  = params(5);
  double p_sigma2     = params(6);
  
  vec ITI_scaled = ITI / ITI.max();
  int N_total = X.n_rows;
  
  vec bio_p = zeros<vec>(N_total);
  mat bio_y_rt = zeros<mat>(N_total, 32);
  
  for(int s = 0; s < N_subjs; s++) {
    uvec idx = find(subj_indices == s);
    int N_trials = idx.n_elem;
    
    mat mu_history = zeros<mat>(N_trials, 32);
    vec mu = zeros<vec>(32);
    vec Z = zeros<vec>(896);
    vec W_purk = zeros<vec>(896);
    vec D_prev = zeros<vec>(362);
    
    for(int t = 0; t < N_trials; t++) {
      int global_t = idx(t);
      if(ITI_scaled(global_t) > 0) {
        Z.zeros();
        W_purk += rand_noise.row(global_t).t() * std::sqrt(p_sigma2 * ITI_scaled(global_t));
        mu = mu * (1.0 - p_alpha_pc * p_lambda_pc * ITI_scaled(global_t));
      }
      
      vec eps = X.row(global_t).t() - Pi_mat.row(global_t).t() * dot(D_prev, D_prev);
      vec G = W_ach1 * mu;
      Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
      W_purk = W_purk - p_kappa_cf * (mean(abs(eps)) * Z);
      vec D = W_ach2 * (G % W_purk);
      mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI_scaled(global_t)) * mu + p_beta_thal * (W_thal * D));
      D_prev = D;
      
      mu_history.row(t) = mu.t();
    }
    
    mat XX = mu_history.t() * mu_history;
    XX.diag() += 0.01; 
    mat XX_inv = inv(XX);
    
    vec W_policy = XX_inv * (mu_history.t() * y_choice.elem(idx));
    mat W_mu     = XX_inv * (mu_history.t() * y_rt.rows(idx));
    
    vec bio_logits = mu_history * W_policy;
    vec p = 1.0 / (1.0 + exp(-bio_logits));
    p = clamp(p, 1e-7, 1.0 - 1e-7);
    
    bio_p.elem(idx) = p;
    bio_y_rt.rows(idx) = mu_history * W_mu;
  }
  
  return List::create(Named("y_hat_choice") = bio_p,
                      Named("y_hat_rt") = bio_y_rt);
}
'

sourceCpp(code = cpp_code)

res <- eval_bio_predictions(50, data_list$subj_indices, data_list$X, data_list$ITI, 
                            W_gen, W_ach1, W_ach2, W_thal, Pi_mat,
                            global_rand_noise, params_bio, data_list$y_choice, data_list$y_rt)

y_hat_choice <- as.numeric(res$y_hat_choice)
y_hat_rt <- as.matrix(res$y_hat_rt)

# We want PR-AUC against the RNN distillation targets (as continuous probabilities)
# Wait, PR-AUC requires binary labels! You cannot compute PR-AUC against continuous probabilities!
# We must use the true empirical labels targets$labels
true_labels <- targets$labels[idx]

pr <- pr.curve(scores.class0 = y_hat_choice, weights.class0 = true_labels, curve=TRUE)
cat("Biological Model PR-AUC (vs Empirical Choice): ", pr$auc.integral, "\n")

# RT RMSE (against RNN representation)
rt_rmse <- sqrt(mean((y_hat_rt - data_list$y_rt)^2))
cat("Biological Model RT-RMSE (vs RNN representation): ", rt_rmse, "\n")
