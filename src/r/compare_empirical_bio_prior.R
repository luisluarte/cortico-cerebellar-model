
library(Rcpp)
library(RcppArmadillo)
library(loo)

cat("Setting up data for Bio-EmpiricalBayes models...\n")
targets <- readRDS("distillation_targets_v2_empirical.rds")
unique_subjs <- unique(targets$subjs)
test_subjs <- unique_subjs[1:50] # In-bag subjects
idx <- which(targets$subjs %in% test_subjs)

data_list <- list(
  N_total = length(idx),
  subj_indices = as.numeric(factor(targets$subjs[idx])) - 1,
  X = as.matrix(targets$X[idx, ]),
  ITI = as.numeric(targets$ITI[idx]),
  lag_reward = as.numeric(targets$lag_reward[idx]),
  lag_ch = as.numeric(targets$lag_ch[idx]),
  lag_resp = as.numeric(targets$lag_resp[idx]),
  y_choice = targets$labels[idx], # TRUE EMPIRICAL CHOICE
  y_rt = targets$true_rt[idx]     # TRUE EMPIRICAL RT
)

eval_q <- function(params, idx, d) {
  alpha_win <- max(1e-7, min(1-1e-7, plogis(params[1])))
  alpha_loss <- max(1e-7, min(1-1e-7, plogis(params[2])))
  beta <- exp(params[3])
  N <- length(idx)
  Q <- c(0.5, 0.5)
  Q_history <- matrix(0, N, 2)
  q_logits <- numeric(N)
  for(t in 1:N) {
    Q_history[t,] <- Q
    ev_left <- Q[1] * d$X[idx[t], 1]
    ev_right <- Q[2] * d$X[idx[t], 2]
    p_bd1 <- plogis(beta * (ev_left - ev_right))
    if(d$lag_resp[idx[t]] == 1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
    p_switch <- max(1e-7, min(1-1e-7, p_switch))
    q_logits[t] <- qlogis(p_switch)
    chosen <- ifelse(d$lag_resp[idx[t]] == 1, 1, 2)
    if(d$lag_reward[idx[t]] > 0) { Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen])
    } else { Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen]) }
  }
  p <- plogis(q_logits)
  bce <- d$y_choice[idx] * log(p + 1e-8) + (1 - d$y_choice[idx]) * log(1 - p + 1e-8)
  XX <- crossprod(Q_history) + diag(0.01, 2)
  XX_inv <- solve(XX)
  W_mu <- XX_inv %*% crossprod(Q_history, d$y_rt[idx])
  bio_mu <- Q_history %*% W_mu
  residuals <- d$y_rt[idx] - bio_mu
  sigma_rt <- sqrt(mean(residuals^2)) + 1e-4
  rt_ll <- numeric(N)
  for(t in 1:N) {
    rt_ll[t] <- -0.5 * log(2.0 * pi * sigma_rt^2) - (residuals[t]^2) / (2.0 * sigma_rt^2)
  }
  return(bce + rt_ll)
}

eval_wsls <- function(params, idx, d) {
  theta_win <- max(1e-7, min(1-1e-7, plogis(params[1])))
  theta_loss <- max(1e-7, min(1-1e-7, plogis(params[2])))
  N <- length(idx)
  wsls_p <- ifelse(d$lag_reward[idx] > 0, theta_win, theta_loss)
  wsls_p[d$lag_reward[idx] == 0 & d$lag_ch[idx] == 0] <- 0.5 
  p <- plogis(qlogis(wsls_p))
  bce <- d$y_choice[idx] * log(p + 1e-8) + (1 - d$y_choice[idx]) * log(1 - p + 1e-8)
  State <- matrix(d$lag_reward[idx], ncol=1)
  XX <- crossprod(State) + diag(0.01, 1)
  XX_inv <- solve(XX)
  W_mu <- XX_inv %*% crossprod(State, d$y_rt[idx])
  bio_mu <- State %*% W_mu
  residuals <- d$y_rt[idx] - bio_mu
  sigma_rt <- sqrt(mean(residuals^2)) + 1e-4
  rt_ll <- numeric(N)
  for(t in 1:N) {
    rt_ll[t] <- -0.5 * log(2.0 * pi * sigma_rt^2) - (residuals[t]^2) / (2.0 * sigma_rt^2)
  }
  return(bce + rt_ll)
}

run_mcmc_unconstrained <- function(eval_fn, N_params, d, iters=500) {
  N_subjs <- length(unique(d$subj_indices))
  mu_pop <- rep(0, N_params)
  sigma_pop <- rep(-1.0, N_params)
  Z_sub <- matrix(rnorm(N_subjs * N_params), N_subjs, N_params)
  log_lik_trace <- matrix(0, iters, d$N_total)
  for(iter in 1:iters) {
    mu_pop_new <- mu_pop + rnorm(N_params) * 0.1
    ll_old <- 0; ll_new <- 0
    for(s in 1:N_subjs) {
      s_idx <- which(d$subj_indices == (s-1))
      ll_old <- ll_old + sum(eval_fn(mu_pop + exp(sigma_pop) * Z_sub[s,], s_idx, d))
      ll_new <- ll_new + sum(eval_fn(mu_pop_new + exp(sigma_pop) * Z_sub[s,], s_idx, d))
    }
    prior_mu_old <- sum(dnorm(mu_pop, 0, 3, log=TRUE))
    prior_mu_new <- sum(dnorm(mu_pop_new, 0, 3, log=TRUE))
    if (log(runif(1)) < (ll_new + prior_mu_new) - (ll_old + prior_mu_old)) {
      mu_pop <- mu_pop_new
    }
    current_ll <- numeric(d$N_total)
    for(s in 1:N_subjs) {
      s_idx <- which(d$subj_indices == (s-1))
      Z_old <- Z_sub[s,]
      Z_new <- Z_old + rnorm(N_params) * 0.1
      res_old <- eval_fn(mu_pop + exp(sigma_pop) * Z_old, s_idx, d)
      res_new <- eval_fn(mu_pop + exp(sigma_pop) * Z_new, s_idx, d)
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

cat("Running Q-Learning MCMC (Unconstrained Empirical)...\n")
q_trace <- run_mcmc_unconstrained(eval_q, 3, data_list, 500)
cat("Running WSLS MCMC (Unconstrained Empirical)...\n")
wsls_trace <- run_mcmc_unconstrained(eval_wsls, 2, data_list, 500)

cat("Running Biological C++ MCMC (Empirical Bayes Prior N(mu_EB, 0.5))...\n")
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

sourceCpp("empirical_mcmc_prior.cpp")
bio_res <- run_empirical_mcmc_prior(500, length(test_subjs), data_list$subj_indices, data_list$X, data_list$ITI, 
                                 data_list$y_choice, data_list$y_rt, Pi_mat,
                                 W_gen, W_ach1, W_ach2, W_thal, data_list$N_total)

burn <- 251:500
bio_trace <- bio_res$log_lik_trace[burn, ]

cat("Computing PSIS-LOO and Comparing (Post Burn-in)...\n")
loo_q <- loo(q_trace[burn, ])
loo_wsls <- loo(wsls_trace[burn, ])
loo_bio <- loo(bio_trace)

cat("\n--- MODEL COMPARISON (Bio+EmpiricalBayes vs Unconstrained Baselines) ---\n")
print(loo_compare(list(Bio = loo_bio, QLearn = loo_q, WSLS = loo_wsls)))
