
library(Rcpp)
library(RcppArmadillo)
library(parallel)
library(loo)
library(posterior)

cat("Starting 04_factorial_final_parallel.R (LONG RUN: 4 chains, 2000 iters)...\n")

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")

if(file.exists("../../results/mu_EB_final.rds")) {
    mu_EB <- readRDS("../../results/mu_EB_final.rds")
} else {
    mu_EB <- readRDS("../../results/mu_EB_static.rds")
    cat("Warning: mu_EB_final.rds not found, using mu_EB_static.rds fallback\n")
}

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

sourceCpp("bio_mcmc_chain.cpp")

# --- Baselines ---
eval_q <- function(params, s_idx, target_mode) {
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
  
  if(target_mode == "empirical") {
    W_rt <- XX_inv %*% crossprod(Q_history, y_rt[s_idx])
    bio_rt <- Q_history %*% W_rt
    res_rt <- y_rt[s_idx] - bio_rt
    sigma_rt <- sqrt(mean(res_rt^2)) + 1e-4
    rt_ll <- -0.5 * log(2.0 * pi * sigma_rt^2) - (res_rt^2) / (2.0 * sigma_rt^2)
    return(sum(bce + rt_ll))
  } else {
    W_dist <- XX_inv %*% crossprod(Q_history, rnn_mu[s_idx,])
    proj_dist <- Q_history %*% W_dist
    res_dist <- rnn_mu[s_idx,] - proj_dist
    sigma_dist <- mean(res_dist^2) + 1e-4
    dist_ll <- rowSums(-0.5 * log(2.0 * pi * sigma_dist) - (res_dist^2) / (2.0 * sigma_dist))
    return(sum(dist_ll))
  }
}

eval_wsls <- function(params, s_idx, target_mode) {
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
  
  if(target_mode == "empirical") {
    W_rt <- XX_inv %*% crossprod(State, y_rt[s_idx])
    bio_rt <- State %*% W_rt
    res_rt <- y_rt[s_idx] - bio_rt
    sigma_rt <- sqrt(mean(res_rt^2)) + 1e-4
    rt_ll <- -0.5 * log(2.0 * pi * sigma_rt^2) - (res_rt^2) / (2.0 * sigma_rt^2)
    return(sum(bce + rt_ll))
  } else {
    W_dist <- XX_inv %*% crossprod(State, rnn_mu[s_idx,])
    proj_dist <- State %*% W_dist
    res_dist <- rnn_mu[s_idx,] - proj_dist
    sigma_dist <- mean(res_dist^2) + 1e-4
    dist_ll <- rowSums(-0.5 * log(2.0 * pi * sigma_dist) - (res_dist^2) / (2.0 * sigma_dist))
    return(sum(dist_ll))
  }
}

run_baseline_chain <- function(eval_fn, N_params, s_idx, target_mode, iters=2000) {
  trace_z <- matrix(0, iters, N_params)
  trace_ll <- matrix(0, iters, length(s_idx))
  Z_curr <- rnorm(N_params)
  ll_curr <- eval_fn(Z_curr, s_idx, target_mode)
  
  for(iter in 1:iters) {
    Z_new <- Z_curr + rnorm(N_params) * 0.05
    ll_new <- eval_fn(Z_new, s_idx, target_mode)
    
    prior_old <- sum(dnorm(Z_curr, 0, 1, log=TRUE))
    prior_new <- sum(dnorm(Z_new, 0, 1, log=TRUE))
    
    if(log(runif(1)) < (ll_new + prior_new) - (ll_curr + prior_old)) {
      Z_curr <- Z_new
      ll_curr <- ll_new
    }
    trace_z[iter, ] <- Z_curr
    trace_ll[iter, ] <- rep(ll_curr / length(s_idx), length(s_idx))
  }
  return(list(trace_z=trace_z, trace_ll=trace_ll))
}

# --- 24 Core Mclapply ---
tasks <- expand.grid(subj_id = 0:(length(subjs)-1), chain = 1:4)
n_cores <- 24
iters <- 2000
burn <- 1001:2000

models <- list(
  list(name="bio_dist", fn="bio", target="distillation", prior="locked"),
  list(name="bio_flat", fn="bio", target="empirical", prior="flat"),
  list(name="bio_lock", fn="bio", target="empirical", prior="locked"),
  list(name="bio_anch", fn="bio", target="empirical", prior="anchored"),
  list(name="q_dist", fn="q", target="distillation"),
  list(name="w_dist", fn="w", target="distillation"),
  list(name="q_emp", fn="q", target="empirical"),
  list(name="w_emp", fn="w", target="empirical")
)

all_results <- list()
# Check if there is an existing incremental save to resume from
if(file.exists("../../results/factorial_final_long_checkpoint.rds")) {
    all_results <- readRDS("../../results/factorial_final_long_checkpoint.rds")
    cat("Resuming from checkpoint!\n")
}

for(m in models) {
  if(m$name %in% names(all_results)) {
    cat(sprintf("Skipping Model: %s (Already completed)\n", m$name))
    next
  }
  
  cat(sprintf("Running Model: %s...\n", m$name))
  
  res <- mclapply(1:nrow(tasks), function(i) {
    s <- tasks$subj_id[i]
    s_idx <- which(subj_indices == s)
    
    if(m$fn == "bio") {
      out <- run_bio_mcmc_chain(iters, X[s_idx,], ITI[s_idx], reward[s_idx], choices[s_idx], y_rt[s_idx], rnn_mu[s_idx,], m$target, m$prior, mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat[s_idx,])
      return(out)
    } else if(m$fn == "q") {
      out <- run_baseline_chain(eval_q, 3, s_idx, m$target, iters)
      return(out)
    } else if(m$fn == "w") {
      out <- run_baseline_chain(eval_wsls, 2, s_idx, m$target, iters)
      return(out)
    }
  }, mc.cores=n_cores)
  
  if(inherits(res[[1]], "try-error")) { print(res[[1]]); stop("mclapply failed") }
  
  # Compute Rhat
  rhats <- numeric(length(subjs))
  ess <- numeric(length(subjs))
  for(s in 0:(length(subjs)-1)) {
    s_tasks <- which(tasks$subj_id == s)
    num_params <- dim(res[[s_tasks[1]]]$trace_z)[2]
    
    arr <- array(0, dim=c(length(burn), 4, num_params))
    for(c in 1:4) {
      arr[, c, ] <- res[[s_tasks[c]]]$trace_z[burn, ]
    }
    
    s_rhats <- numeric(num_params)
    s_ess <- numeric(num_params)
    for(p in 1:num_params) {
      s_rhats[p] <- posterior::rhat(arr[,,p])
      s_ess[p] <- posterior::ess_bulk(arr[,,p])
    }
    rhats[s+1] <- max(s_rhats, na.rm=TRUE) 
    ess[s+1] <- min(s_ess, na.rm=TRUE)     
  }
  
  # Compute pooled LL trace for LOO
  total_ll_trace <- matrix(0, length(burn)*4, nrow(d))
  trace_row <- 1
  for(c in 1:4) {
    c_tasks <- which(tasks$chain == c)
    for(iter in burn) {
      for(i in c_tasks) {
        s <- tasks$subj_id[i]
        s_idx <- which(subj_indices == s)
        total_ll_trace[trace_row, s_idx] <- res[[i]]$trace_ll[iter, ]
      }
      trace_row <- trace_row + 1
    }
  }
  
  loo_obj <- loo(total_ll_trace)
  
  all_results[[m$name]] <- list(
    loo = loo_obj,
    rhats = rhats,
    ess = ess
  )
  
  # Incremental Save
  saveRDS(all_results, "../../results/factorial_final_long_checkpoint.rds")
}

saveRDS(all_results, "../../results/factorial_final_long_A.rds")
cat("Finished! Saved factorial_final_long_A.rds\n")
