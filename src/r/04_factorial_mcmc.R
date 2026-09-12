
library(Rcpp)
library(RcppArmadillo)
library(loo)
library(parallel)

cat("Starting 04_factorial_mcmc.R...\n")
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

run_experiment <- function(dataset_name) {
  d <- readRDS(paste0("../../data/", dataset_name, ".rds"))
  targets <- readRDS("../../results/oracle_targets.rds")
  mu_EB <- readRDS("../../results/mu_EB.rds")
  
  subjs <- unique(d$participant_id)
  
  X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled")])
  ITI <- d$ITI_scaled
  reward <- d$lag_Reward
  choices <- d$Resp_mapped
  
  # Fetch dimension of the target hidden layer dynamically
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
  
  run_chains <- function(target, prior) {
    cat(sprintf("   Launching 3 chains for Target: %s | Prior: %s\n", target, prior))
    res <- mclapply(1:3, function(chain) {
      run_bio_mcmc(500, length(subjs), subj_indices, X, ITI, reward, choices, rnn_logits, rnn_mu, nrow(d), target, prior, mu_EB)
    }, mc.cores=3)
    
    rhats <- numeric(7)
    for(p in 1:7) {
       trace_mat <- cbind(res[[1]]$trace_mu[251:500, p],
                          res[[2]]$trace_mu[251:500, p],
                          res[[3]]$trace_mu[251:500, p])
       rhats[p] <- calc_rhat(trace_mat)
    }
    cat("   Rhats for 7 parameters: ", round(rhats, 3), "\n")
    
    ll_mat <- matrix(0, nrow=750, ncol=nrow(d))
    for(c in 1:3) {
      ll_mat[((c-1)*250 + 1):(c*250), ] <- res[[c]]$log_lik_trace[251:500, ]
    }
    loo_obj <- loo(ll_mat)
    
    return(list(loo = loo_obj, rhats = rhats))
  }
  
  cat("\n--- Running Experiment 1: Distillation on", dataset_name, "---\n")
  exp1 <- run_chains("distillation", "locked")
  
  cat("\n--- Running Experiment 2: Chaotic Empirical on", dataset_name, "---\n")
  exp2 <- run_chains("empirical", "flat")
  
  cat("\n--- Running Experiment 3: Zero-Shot Empirical on", dataset_name, "---\n")
  exp3 <- run_chains("empirical", "locked")
  
  cat("\n--- Running Experiment 4: Guided Empirical Bayes on", dataset_name, "---\n")
  exp4 <- run_chains("empirical", "anchored")
  
  saveRDS(list(exp1=exp1, exp2=exp2, exp3=exp3, exp4=exp4), 
          paste0("../../results/factorial_", dataset_name, "_results.rds"))
}

run_experiment("dataset_A")
run_experiment("dataset_B")

cat("Factorial evaluation completed successfully!\n")
