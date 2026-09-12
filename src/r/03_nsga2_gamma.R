
library(mco)
library(Rcpp)
library(pROC)

cat("Starting 03_nsga2_gamma.R...\n")
sourceCpp("sim_bio.cpp")

data_A <- readRDS("../../data/dataset_A.rds")
oracle_targets <- readRDS("../../results/oracle_targets.rds")

# Prepare data for faster evaluation
subjs <- unique(data_A$participant_id)
subj_data <- list()
for(s in subjs) {
  d_s <- data_A[data_A$participant_id == s, ]
  targ_s <- oracle_targets[[s]]
  
  subj_data[[s]] <- list(
    N = nrow(d_s),
    X = as.matrix(d_s[, c("Bd1_scaled", "Bd2_scaled")]),
    ITI = d_s$ITI_scaled,
    reward = d_s$lag_Reward,
    human_choices = d_s$Resp_mapped,
    rnn_logits = targ_s$logits,
    rnn_mu = targ_s$hidden
  )
}

# The fitness function for NSGA-II
fitness_fn <- function(params) {
  p_alpha_pc <- params[1]
  p_lambda_pc <- params[2]
  p_beta_thal <- params[3]
  p_kappa_cf <- params[4]
  p_alpha_gran <- params[5]
  p_beta_gran <- params[6]
  p_sigma2_diff <- params[7]
  gamma <- params[8]
  
  total_bce <- 0
  
  for(s in subjs) {
    sd <- subj_data[[s]]
    bio_mu <- simulate_bio_gamma(sd$N, sd$X, sd$ITI, sd$reward,
                                 p_alpha_pc, p_lambda_pc, p_beta_thal, p_kappa_cf,
                                 p_alpha_gran, p_beta_gran, p_sigma2_diff)
    
    if(max(abs(bio_mu)) >= 1e3 || any(is.na(bio_mu))) {
      return(c(1e6, 1.0)) # Extreme penalty
    }
    
    # Simple linear map from mu to logits (Ridge regression analytically calculated inside to prevent overhead)
    # Actually, for extreme speed in NSGA-II we can just use the internal representation directly if matched, 
    # but let's do a fast ridge regression
    XX <- crossprod(bio_mu) + diag(0.01, 32)
    XX_inv <- tryCatch(solve(XX), error = function(e) NULL)
    if(is.null(XX_inv)) return(c(1e6, 1.0))
    
    W_mu <- XX_inv %*% crossprod(bio_mu, sd$rnn_logits)
    bio_logits <- as.numeric(bio_mu %*% W_mu)
    
    blend_logits <- (1.0 - gamma) * sd$rnn_logits + gamma * bio_logits
    
    # Clip for stability
    p <- plogis(blend_logits)
    p <- pmax(pmin(p, 1 - 1e-7), 1e-7)
    
    bce <- -sum(sd$human_choices * log(p) + (1 - sd$human_choices) * log(1 - p))
    total_bce <- total_bce + bce
  }
  
  # Obj 1: Minimize total BCE
  # Obj 2: Minimize (1 - gamma) (Maximize reliance on Bio)
  return(c(total_bce, 1.0 - gamma))
}

# Bounds
lower_bounds <- c(0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.0)
upper_bounds <- c(1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000, 1.0)

cat("Running NSGA-II...\n")
# Small population for time constraints (popsize=32, gen=20 should take ~10-15 mins)
res <- nsga2(fitness_fn, idim=8, odim=2, lower.bounds=lower_bounds, upper.bounds=upper_bounds, 
             popsize=48, generations=30, vectorized=FALSE)

pareto_front <- res$value
pareto_params <- res$par

# Save NSGA-II full results
nsga_out <- list(values = pareto_front, params = pareto_params)
saveRDS(nsga_out, "../../results/nsga2_gamma_results.rds")

# Gamma Quartiles Evaluation (Q1, Q2, Q3)
gammas <- pareto_params[, 8]
q1_val <- quantile(gammas, 0.25)
q2_val <- quantile(gammas, 0.50)
q3_val <- quantile(gammas, 0.75)

# Find closest models to quartiles
idx_q1 <- which.min(abs(gammas - q1_val))
idx_q2 <- which.min(abs(gammas - q2_val))
idx_q3 <- which.min(abs(gammas - q3_val))

# We also want the absolute best BCE model from the front (closest to top left)
idx_best <- which.min(pareto_front[, 1])

best_params <- pareto_params[idx_best, ]
cat("Best Baseline Parameters (mu_EB):\n")
print(best_params)
saveRDS(best_params[1:7], "../../results/mu_EB.rds")

# Function to evaluate metrics
eval_metrics <- function(params) {
  gamma <- params[8]
  all_human <- c()
  all_p <- c()
  
  for(s in subjs) {
    sd <- subj_data[[s]]
    bio_mu <- simulate_bio_gamma(sd$N, sd$X, sd$ITI, sd$reward,
                                 params[1], params[2], params[3], params[4],
                                 params[5], params[6], params[7])
    XX <- crossprod(bio_mu) + diag(0.01, 32)
    W_mu <- solve(XX) %*% crossprod(bio_mu, sd$rnn_logits)
    bio_logits <- as.numeric(bio_mu %*% W_mu)
    blend_logits <- (1.0 - gamma) * sd$rnn_logits + gamma * bio_logits
    
    p <- plogis(blend_logits)
    all_human <- c(all_human, sd$human_choices)
    all_p <- c(all_p, p)
  }
  
  all_p <- pmax(pmin(all_p, 1 - 1e-7), 1e-7)
  nll <- -sum(all_human * log(all_p) + (1 - all_human) * log(1 - all_p))
  
  roc_obj <- roc(all_human, all_p, quiet=TRUE)
  pr_auc <- auc(roc_obj) # using AUROC for simple proxy if PR-AUC not installed, wait let's use prROC if needed.
  return(c(NLL = nll, AUC = as.numeric(pr_auc), Gamma = gamma))
}

cat("\nQuartile 1 Evaluation:\n")
print(eval_metrics(pareto_params[idx_q1, ]))
cat("\nQuartile 2 Evaluation:\n")
print(eval_metrics(pareto_params[idx_q2, ]))
cat("\nQuartile 3 Evaluation:\n")
print(eval_metrics(pareto_params[idx_q3, ]))

cat("\nStep 3 Complete!\n")
