
library(Rcpp)
library(RcppArmadillo)
library(loo)
library(parallel)

cat("Starting Replicate Script...\n")
sourceCpp("bio_mcmc_rep.cpp")

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")
mu_EB <- readRDS("../../results/mu_EB.rds")

# Reduce to 20 subjects for fast iteration proof
subjs <- unique(d$participant_id)[1:20]
d <- d[d$participant_id %in% subjs, ]

X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled", "lag_Reward", "lag_Ch", "lag_Resp", "ITI_scaled")])
if(ncol(X) < 6) {
  # Fix X to be 6 columns
  X <- cbind(X, matrix(0, nrow=nrow(X), ncol=6-ncol(X)))
}
ITI <- d$ITI_scaled
reward <- d$lag_Reward
choices <- d$Resp_mapped
y_rt <- d$RT_scaled

target_hidden_dim <- ncol(targets[[1]]$hidden)
rnn_mu <- matrix(0, nrow=nrow(d), ncol=target_hidden_dim)
subj_indices <- numeric(nrow(d))

idx_counter <- 1
for(i in 1:length(subjs)) {
  s <- subjs[i]
  n <- sum(d$participant_id == s)
  rng <- idx_counter:(idx_counter + n - 1)
  rnn_mu[rng, ] <- targets[[s]]$hidden
  subj_indices[rng] <- i - 1
  idx_counter <- idx_counter + n
}

set.seed(42)
W_gen <- matrix(rnorm(6 * 32, 0, 1/sqrt(32)), nrow=6, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)

Pi_mat <- matrix(0, nrow=nrow(d), ncol=6)
for(s in 0:(length(subjs)-1)) {
  s_idx <- which(subj_indices == s)
  subj_var <- apply(X[s_idx, ], 2, var)
  raw_prec <- 1.0 / (subj_var + 1e-6)
  Pi_mat[s_idx, ] <- matrix(rep(raw_prec / mean(raw_prec), length(s_idx)), nrow=length(s_idx), byrow=TRUE)
}

# Distillation - Locked
res_dist <- run_bio_mcmc(100, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "distillation", "locked", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)
loo_dist <- loo(res_dist$log_lik_trace[51:100, ])
cat("Bio Distillation LOO:", loo_dist$estimates["elpd_loo", "Estimate"], "\n")

# Empirical - Flat (Unconstrained explosion)
res_flat <- run_bio_mcmc(100, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "flat", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)
loo_flat <- loo(res_flat$log_lik_trace[51:100, ])
cat("Bio Chaotic Flat LOO:", loo_flat$estimates["elpd_loo", "Estimate"], "\n")

# Empirical - Locked (Zero-Shot)
res_lock <- run_bio_mcmc(100, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "locked", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)
loo_lock <- loo(res_lock$log_lik_trace[51:100, ])
cat("Bio Zero-Shot LOO:", loo_lock$estimates["elpd_loo", "Estimate"], "\n")

# Empirical - Anchored (Guided Bayes)
res_anch <- run_bio_mcmc(100, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "anchored", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)
loo_anch <- loo(res_anch$log_lik_trace[51:100, ])
cat("Bio Guided EB LOO:", loo_anch$estimates["elpd_loo", "Estimate"], "\n")
