local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
  library(mco)
  library(doParallel)
  library(foreach)
  library(tibble)
  library(Rcpp)
  library(RcppArmadillo)
})

cat("Loading Distillation Targets 2.0...\n")
targets <- readRDS("distillation_targets_v2.rds")

# Global Platt Scaling Calibration
cat("Calibrating the Teacher's Logits via Platt Scaling...\n")
fit <- glm(targets$labels ~ targets$rnn_logits, family = binomial)
calibrated_rnn_logits <- predict(fit, type = "link")
teacher_soft_targets <- plogis(calibrated_rnn_logits)

# Extract elements
N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
ITI <- ITI / max(ITI)
lag_Reward <- targets$lag_reward
lag_Ch <- targets$lag_ch
lag_Resp <- targets$lag_resp

# The Exact Distillation Targets
rnn_p_logits <- calibrated_rnn_logits
rnn_mu <- targets$rnn_mu
rnn_sigma <- targets$rnn_sigma

# Max Capacity Bio Initialization
set.seed(42)
input_dim <- ncol(X)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1/sqrt(32)), nrow=input_dim, ncol=32)
W_ach1 <- matrix(rnorm(2000 * 32, 0, 1/sqrt(32)), nrow=2000, ncol=32)
W_ach2 <- matrix(rnorm(1000 * 2000, 0, 1/sqrt(2000)), nrow=1000, ncol=2000)
W_thal <- matrix(rnorm(32 * 1000, 0, 1/sqrt(1000)), nrow=32, ncol=1000)

Pi_mat <- matrix(0, nrow=N_trials, ncol=input_dim)
unique_subjs <- unique(targets$subjs)
for(s in unique_subjs) {
    idx <- which(targets$subjs == s)
    subj_X <- targets$X[idx, ]
    subj_var <- apply(subj_X, 2, var)
    raw_prec <- 1.0 / (subj_var + 1e-6)
    norm_prec <- raw_prec / mean(raw_prec)
    Pi_mat[idx, ] <- matrix(rep(norm_prec, length(idx)), nrow=length(idx), byrow=TRUE)
}

# Distillation Hyperparameter (Gamma)
gamma <- 0.8  # Heavy reliance on Teacher representations to provide structural tension

# --- FAST PRUNING NSGA-II SETTINGS ---
POPSIZE <- 40
GENERATIONS <- 20

# Run on Dataset A (Subsample for rapid pruning test)
train_subjs <- unique_subjs[1:50] # Top 50 subjects (Dataset A)
train_idx <- which(targets$subjs %in% train_subjs)

t_N <- length(train_idx)
t_X <- X[train_idx, ]
t_ITI <- ITI[train_idx]
t_subjs <- targets$subjs[train_idx]
t_rnn_logits <- calibrated_rnn_logits[train_idx]
t_rnn_mu <- targets$rnn_mu[train_idx, ]

num_cores <- detectCores()
cat(sprintf("Starting Parallel Cluster on %d Cores...\n", num_cores))
cl <- makeCluster(num_cores)
registerDoParallel(cl)

cat("Compiling C++ Pruning Simulator on all Cores...\n")
clusterEvalQ(cl, {
  library(Rcpp)
  library(RcppArmadillo)
  sourceCpp("sim_bio_pruning.cpp")
})

cat("Starting Biological Evolutionary Pruning NSGA-II...\n")

# Objective function for Bio Pruning
bio_pruning_obj <- function(params_matrix) {
  if (is.null(nrow(params_matrix))) params_matrix <- matrix(params_matrix, ncol = 1)
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  
  results <- foreach(ind = 1:num_ind, .combine = cbind, .export=c("t_X", "t_ITI", "t_subjs", "t_rnn_mu", "W_gen", "W_ach1", "W_ach2", "W_thal", "Pi_mat", "train_subjs", "gamma")) %dopar% {
    
    # 7 Bio Parameters
    p_alpha_pc   <- max(1e-5, min(0.99, params_matrix[1, ind]))
    p_lambda_pc  <- max(1e-5, min(0.99, params_matrix[2, ind]))
    p_beta_thal  <- max(1e-5, min(0.99, params_matrix[3, ind]))
    p_kappa_cf   <- max(1e-5, min(0.99, params_matrix[4, ind]))
    p_alpha_gran <- max(1e-5, min(0.99, params_matrix[5, ind]))
    p_beta_gran  <- max(1e-5, min(0.99, params_matrix[6, ind]))
    p_sigma2     <- max(1e-5, min(5.0,  params_matrix[7, ind]))
    
    # 2 Pruning Hyper-parameters (Ratios)
    p_gran_ratio <- max(0.01, min(1.0, params_matrix[8, ind]))
    p_dcn_ratio  <- max(0.01, min(1.0, params_matrix[9, ind]))
    
    total_mu_mse <- 0.0
    valid_count <- 0
    
    for(s in train_subjs) {
        idx <- which(t_subjs == s)
        subj_X <- t_X[idx, , drop=FALSE]
        subj_ITI <- t_ITI[idx]
        subj_Pi <- Pi_mat[idx, , drop=FALSE]
        
        sim_mu <- simulate_bio_pruning(length(idx), subj_X, subj_ITI, W_gen, W_ach1, W_ach2, W_thal, subj_Pi, 
                               p_alpha_pc, p_lambda_pc, p_beta_thal, p_kappa_cf, p_alpha_gran, p_beta_gran, p_sigma2,
                               p_gran_ratio, p_dcn_ratio)
                               
        if(length(sim_mu) == 1 && is.na(sim_mu[1,1])) {
            total_mu_mse <- 9999.0
            break
        }
        
        target_mu <- t_rnn_mu[idx, , drop=FALSE]
        total_mu_mse <- total_mu_mse + mean((sim_mu - target_mu)^2)
        valid_count <- valid_count + 1
    }
    
    if(total_mu_mse == 9999.0) {
        distillation_loss <- 9999.0
    } else {
        distillation_loss <- total_mu_mse / valid_count
    }
    
    # Objective 2: Structural Network Cost (Rent's Rule Approximation)
    # Minimizing the survival ratios mathematically forces pruning
    structural_cost <- p_gran_ratio + p_dcn_ratio
    
    c(distillation_loss, structural_cost)
  }
  
  return(results)
}

# 7 Bio Parameters + 2 Pruning Parameters
lower_bounds <- c(rep(1e-4, 6), 1e-3, 0.01, 0.01)
upper_bounds <- c(rep(0.99, 6), 2.0, 1.0, 1.0)

res <- nsga2(bio_pruning_obj, idim = 9, odim = 2,
             lower.bounds = lower_bounds,
             upper.bounds = upper_bounds,
             popsize = POPSIZE,
             generations = GENERATIONS,
             vectorized = TRUE)

saveRDS(res, "../../results/nsga2_pruning_results.rds")
cat("Pruning NSGA-II Complete! Saved to ../../results/nsga2_pruning_results.rds\n")
stopCluster(cl)
