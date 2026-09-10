
library(Rcpp)
library(RcppArmadillo)
library(parallel)
library(loo)
library(posterior)

cat("Starting 07_cov_test.R (bio_dist only, Empirical Covariance Adaptation)...\n")

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")

if(file.exists("../../results/mu_EB_final.rds")) {
    mu_EB <- readRDS("../../results/mu_EB_final.rds")
} else {
    mu_EB <- readRDS("../../results/mu_EB_static.rds")
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

sourceCpp("bio_mcmc_chain_cov.cpp")

tasks <- expand.grid(subj_id = 0:(length(subjs)-1), chain = 1:4)
n_cores <- 24
iters <- 2000
burn <- 1001:2000

res <- mclapply(1:nrow(tasks), function(i) {
  s <- tasks$subj_id[i]
  s_idx <- which(subj_indices == s)
  out <- run_bio_mcmc_chain_cov(iters, X[s_idx,], ITI[s_idx], reward[s_idx], choices[s_idx], y_rt[s_idx], rnn_mu[s_idx,], "distillation", "locked", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat[s_idx,])
  return(out)
}, mc.cores=n_cores)

if(inherits(res[[1]], "try-error")) { print(res[[1]]); stop("mclapply failed") }

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

cat(sprintf("\n--- FINAL RESULTS (bio_dist with Covariance Adaptation) ---\nELPD: %.1f\nMin Rhat: %.3f | Median Rhat: %.3f | Mean Rhat: %.3f | Max Rhat: %.3f\nMin ESS: %.1f\n", 
            loo_obj$estimates["elpd_loo", "Estimate"], 
            min(rhats, na.rm=TRUE), median(rhats, na.rm=TRUE), mean(rhats, na.rm=TRUE), max(rhats, na.rm=TRUE), 
            min(ess, na.rm=TRUE)))
