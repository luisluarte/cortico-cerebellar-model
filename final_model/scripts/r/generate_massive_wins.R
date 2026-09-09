
library(Rcpp)
library(RcppArmadillo)
library(loo)
library(parallel)

cat("Starting Replicate Massive Wins script...\n")

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")
mu_EB <- readRDS("../../results/mu_EB_static.rds") # Use STATIC!

X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled", "lag_Reward", "lag_Ch", "lag_Resp", "ITI_scaled")])
if(ncol(X) < 6) X <- cbind(X, matrix(0, nrow=nrow(X), ncol=6-ncol(X)))
ITI <- d$ITI_scaled
reward <- d$lag_Reward
choices <- d$Resp_mapped
y_rt <- d$RT_scaled

subjs <- unique(d$participant_id)
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

sourceCpp("bio_mcmc_rep.cpp")

# --- Bio Runs ---
cat("Running Bio Distillation (Locked)...\n")
res_bio_dist <- run_bio_mcmc(200, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "distillation", "locked", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)

cat("Running Bio Chaotic (Flat)...\n")
res_bio_flat <- run_bio_mcmc(200, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "flat", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)

cat("Running Bio Zero-Shot (Locked)...\n")
res_bio_lock <- run_bio_mcmc(200, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "locked", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)

cat("Running Bio Guided (Anchored)...\n")
res_bio_anch <- run_bio_mcmc(200, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, nrow(d), "empirical", "anchored", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)


# --- Baseline Runs ---
eval_q <- function(params, s_idx) {
  alpha_win <- max(1e-7, min(1-1e-7, plogis(params[1])))
  alpha_loss <- max(1e-7, min(1-1e-7, plogis(params[2])))
  beta <- exp(params[3])
  N <- length(s_idx)
  Q <- c(0.5, 0.5)
  Q_history <- matrix(0, N, 2)
  q_logits <- numeric(N)
  for(t in 1:N) {
    Q_history[t,] <- Q
    ev_left <- Q[1] * X[s_idx[t], 1]
    ev_right <- Q[2] * X[s_idx[t], 2]
    p_bd1 <- plogis(beta * (ev_left - ev_right))
    if(X[s_idx[t], 5] == 1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
    p_switch <- max(1e-7, min(1-1e-7, p_switch))
    q_logits[t] <- qlogis(p_switch)
    chosen <- ifelse(X[s_idx[t], 5] == 1, 1, 2)
    if(reward[s_idx[t]] > 0) Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen])
    else Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen])
  }
  p <- plogis(q_logits)
  bce <- choices[s_idx] * log(p + 1e-8) + (1 - choices[s_idx]) * log(1 - p + 1e-8)
  
  XX <- crossprod(Q_history) + diag(0.01, 2)
  XX_inv <- solve(XX)
  
  # Empirical Readout
  W_rt <- XX_inv %*% crossprod(Q_history, y_rt[s_idx])
  bio_rt <- Q_history %*% W_rt
  res_rt <- y_rt[s_idx] - bio_rt
  sigma_rt <- sqrt(mean(res_rt^2)) + 1e-4
  rt_ll <- -0.5 * log(2.0 * pi * sigma_rt^2) - (res_rt^2) / (2.0 * sigma_rt^2)
  
  # Distillation Readout
  W_dist <- XX_inv %*% crossprod(Q_history, rnn_mu[s_idx,])
  proj_dist <- Q_history %*% W_dist
  res_dist <- rnn_mu[s_idx,] - proj_dist
  sigma_dist <- mean(res_dist^2) + 1e-4
  dist_ll <- rowSums(-0.5 * log(2.0 * pi * sigma_dist) - (res_dist^2) / (2.0 * sigma_dist))
  
  return(list(emp_ll = bce + rt_ll, dist_ll = dist_ll))
}

eval_wsls <- function(params, s_idx) {
  theta_win <- max(1e-7, min(1-1e-7, plogis(params[1])))
  theta_loss <- max(1e-7, min(1-1e-7, plogis(params[2])))
  N <- length(s_idx)
  wsls_p <- ifelse(reward[s_idx] > 0, theta_win, theta_loss)
  wsls_p[reward[s_idx] == 0 & X[s_idx, 4] == 0] <- 0.5 
  p <- plogis(qlogis(wsls_p))
  bce <- choices[s_idx] * log(p + 1e-8) + (1 - choices[s_idx]) * log(1 - p + 1e-8)
  
  State <- matrix(reward[s_idx], ncol=1)
  XX <- crossprod(State) + diag(0.01, 1)
  XX_inv <- solve(XX)
  
  W_rt <- XX_inv %*% crossprod(State, y_rt[s_idx])
  bio_rt <- State %*% W_rt
  res_rt <- y_rt[s_idx] - bio_rt
  sigma_rt <- sqrt(mean(res_rt^2)) + 1e-4
  rt_ll <- -0.5 * log(2.0 * pi * sigma_rt^2) - (res_rt^2) / (2.0 * sigma_rt^2)
  
  W_dist <- XX_inv %*% crossprod(State, rnn_mu[s_idx,])
  proj_dist <- State %*% W_dist
  res_dist <- rnn_mu[s_idx,] - proj_dist
  sigma_dist <- mean(res_dist^2) + 1e-4
  dist_ll <- rowSums(-0.5 * log(2.0 * pi * sigma_dist) - (res_dist^2) / (2.0 * sigma_dist))
  
  return(list(emp_ll = bce + rt_ll, dist_ll = dist_ll))
}

run_baseline_mcmc <- function(eval_fn, N_params, target_mode="empirical") {
  mu_pop <- rep(0, N_params)
  sigma_pop <- rep(-3, N_params)
  Z_sub <- matrix(rnorm(length(subjs) * N_params), length(subjs), N_params)
  log_lik_trace <- matrix(0, 200, nrow(d))
  
  for(iter in 1:200) {
    current_ll <- numeric(nrow(d))
    for(s in 1:length(subjs)) {
      s_idx <- which(subj_indices == (s-1))
      Z_old <- Z_sub[s,]
      Z_new <- Z_old + rnorm(N_params) * 0.1
      
      out_old <- eval_fn(mu_pop + exp(sigma_pop) * Z_old, s_idx)
      out_new <- eval_fn(mu_pop + exp(sigma_pop) * Z_new, s_idx)
      
      res_old <- if(target_mode=="distillation") out_old$dist_ll else out_old$emp_ll
      res_new <- if(target_mode=="distillation") out_new$dist_ll else out_new$emp_ll
      
      prior_old <- sum(dnorm(Z_old, 0, 1, log=TRUE))
      prior_new <- sum(dnorm(Z_new, 0, 1, log=TRUE))
      
      if(log(runif(1)) < (sum(res_new) + prior_new) - (sum(res_old) + prior_old)) {
        Z_sub[s,] <- Z_new
        current_ll[s_idx] <- res_new
      } else {
        current_ll[s_idx] <- res_old
      }
    }
    log_lik_trace[iter, ] <- current_ll
  }
  return(log_lik_trace)
}

cat("Running QLearn Distillation...\n")
res_q_dist <- run_baseline_mcmc(eval_q, 3, "distillation")
cat("Running WSLS Distillation...\n")
res_w_dist <- run_baseline_mcmc(eval_wsls, 2, "distillation")

cat("Running QLearn Empirical...\n")
res_q_emp <- run_baseline_mcmc(eval_q, 3, "empirical")
cat("Running WSLS Empirical...\n")
res_w_emp <- run_baseline_mcmc(eval_wsls, 2, "empirical")

# 100 burn-in
burn <- 101:200

suppressWarnings({
  loo_bio_dist <- loo(res_bio_dist$log_lik_trace[burn, ])
  loo_q_dist <- loo(res_q_dist[burn, ])
  loo_w_dist <- loo(res_w_dist[burn, ])
  
  loo_bio_flat <- loo(res_bio_flat$log_lik_trace[burn, ])
  loo_bio_lock <- loo(res_bio_lock$log_lik_trace[burn, ])
  loo_bio_anch <- loo(res_bio_anch$log_lik_trace[burn, ])
  loo_q_emp <- loo(res_q_emp[burn, ])
  loo_w_emp <- loo(res_w_emp[burn, ])
})

cat("\n================ MASSIVE WINS REPORT =================\n")
cat("EXPERIMENT 1: DISTILLATION (Target: RNN Latents)\n")
cat("Bio (Locked):", loo_bio_dist$estimates["elpd_loo", "Estimate"], "\n")
cat("Q-Learn:", loo_q_dist$estimates["elpd_loo", "Estimate"], "\n")
cat("WSLS:", loo_w_dist$estimates["elpd_loo", "Estimate"], "\n")
cat("\n")
cat("EXPERIMENT 2,3,4: EMPIRICAL (Target: Choice + RT)\n")
cat("Q-Learn (Flat):", loo_q_emp$estimates["elpd_loo", "Estimate"], "\n")
cat("WSLS (Flat):", loo_w_emp$estimates["elpd_loo", "Estimate"], "\n")
cat("Bio Exp 2 (Flat / Chaotic):", loo_bio_flat$estimates["elpd_loo", "Estimate"], "\n")
cat("Bio Exp 3 (Locked / Zero-Shot):", loo_bio_lock$estimates["elpd_loo", "Estimate"], "\n")
cat("Bio Exp 4 (Anchored / Guided):", loo_bio_anch$estimates["elpd_loo", "Estimate"], "\n")
cat("=======================================================\n")

cat("Saving all .rds files...\n")

saveRDS(list(
  loo_bio_dist = loo_bio_dist,
  loo_q_dist = loo_q_dist,
  loo_w_dist = loo_w_dist
), "../../results/experiment1_distillation_loo.rds")

saveRDS(list(
  loo_bio_flat = loo_bio_flat,
  loo_bio_lock = loo_bio_lock,
  loo_bio_anch = loo_bio_anch,
  loo_q_emp = loo_q_emp,
  loo_w_emp = loo_w_emp
), "../../results/experiment234_empirical_loo.rds")

saveRDS(list(
  res_bio_dist = res_bio_dist$log_lik_trace,
  res_bio_flat = res_bio_flat$log_lik_trace,
  res_bio_lock = res_bio_lock$log_lik_trace,
  res_bio_anch = res_bio_anch$log_lik_trace,
  res_q_dist = res_q_dist,
  res_w_dist = res_w_dist,
  res_q_emp = res_q_emp,
  res_w_emp = res_w_emp
), "../../results/full_mcmc_traces_A.rds")

cat("All results successfully saved to .rds!\n")
