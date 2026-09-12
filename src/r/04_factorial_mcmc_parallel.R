
library(Rcpp)
library(RcppArmadillo)
library(loo)
library(parallel)

cat("Starting 04_factorial_mcmc_parallel.R...\n")
sourceCpp("bio_mcmc.cpp")

calc_rhat <- function(trace_mat) {
  n <- nrow(trace_mat)
  m <- ncol(trace_mat)
  if(m < 2) return(1)
  chain_means <- colMeans(trace_mat)
  global_mean <- mean(chain_means)
  B <- n / (m - 1) * sum((chain_means - global_mean)^2)
  W <- mean(apply(trace_mat, 2, var))
  if (W == 0) return(1)
  var_plus <- (n - 1) / n * W + B / n
  return(sqrt(var_plus / W))
}

grid <- expand.grid(
  chain = 1:3,
  exp_name = c("dist", "emp_flat", "emp_lock", "emp_anch"),
  dataset = c("dataset_A", "dataset_B"),
  stringsAsFactors = FALSE
)

tasks <- lapply(1:nrow(grid), function(i) grid[i, ])

worker <- function(task) {
  d_name <- task$dataset
  e_name <- task$exp_name
  
  if(e_name == "dist") { target <- "distillation"; prior <- "locked" }
  if(e_name == "emp_flat") { target <- "empirical"; prior <- "flat" }
  if(e_name == "emp_lock") { target <- "empirical"; prior <- "locked" }
  if(e_name == "emp_anch") { target <- "empirical"; prior <- "anchored" }
  
  d <- readRDS(paste0("../../data/", d_name, ".rds"))
  targets <- readRDS("../../results/oracle_targets.rds")
  mu_EB <- readRDS("../../results/mu_EB.rds")
  
  subjs <- unique(d$participant_id)
  X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled")])
  ITI <- d$ITI_scaled
  reward <- d$lag_Reward
  choices <- d$Resp_mapped
  
  target_hidden_dim <- ncol(targets[[1]]$hidden)
  rnn_logits <- numeric(nrow(d))
  rnn_mu <- matrix(0, nrow=nrow(d), ncol=target_hidden_dim)
  subj_indices <- numeric(nrow(d))
  
  idx_counter <- 1
  for(i in 1:length(subjs)) {
    s <- subjs[i]
    n <- sum(d$participant_id == s)
    rng <- idx_counter:(idx_counter + n - 1)
    
    if(!is.null(targets[[s]])) {
      rnn_logits[rng] <- targets[[s]]$logits
      rnn_mu[rng, ] <- targets[[s]]$hidden
    }
    subj_indices[rng] <- i - 1
    idx_counter <- idx_counter + n
  }
  
  res <- run_bio_mcmc(500, length(subjs), subj_indices, X, ITI, reward, choices, rnn_logits, rnn_mu, nrow(d), target, prior, mu_EB)
  
  return(list(task=task, res=res, nrow_d=nrow(d)))
}

cat("Executing across 24 cores... This might take an hour or two depending on the C++ overhead.\n")
raw_results <- mclapply(tasks, worker, mc.cores = 24)

cat("Reassembling results...\n")

for(d_name in c("dataset_A", "dataset_B")) {
  d_out <- list()
  for(e_name in c("dist", "emp_flat", "emp_lock", "emp_anch")) {
    
    chains <- list()
    nrow_d <- 0
    for(r in raw_results) {
      if(!inherits(r, "try-error") && r$task$dataset == d_name && r$task$exp_name == e_name) {
        chains[[r$task$chain]] <- r$res
        nrow_d <- r$nrow_d
      }
    }
    
    if(length(chains) == 3) {
      rhats <- numeric(7)
      for(p in 1:7) {
        trace_mat <- cbind(chains[[1]]$trace_mu[251:500, p],
                           chains[[2]]$trace_mu[251:500, p],
                           chains[[3]]$trace_mu[251:500, p])
        rhats[p] <- calc_rhat(trace_mat)
      }
      
      ll_mat <- matrix(0, nrow=750, ncol=nrow_d)
      for(c in 1:3) {
        ll_mat[((c-1)*250 + 1):(c*250), ] <- chains[[c]]$log_lik_trace[251:500, ]
      }
      
      # Handle pareto-k warnings silently
      loo_obj <- suppressWarnings(loo(ll_mat))
      d_out[[e_name]] <- list(loo = loo_obj, rhats = rhats)
    } else {
      cat("Missing chains for", d_name, e_name, "\n")
      # Print error
      for(r in raw_results) {
        if(inherits(r, "try-error")) {
          print(r)
        }
      }
    }
  }
  saveRDS(d_out, paste0("../../results/factorial_", d_name, "_results.rds"))
}

cat("Factorial evaluation completed successfully!\n")
