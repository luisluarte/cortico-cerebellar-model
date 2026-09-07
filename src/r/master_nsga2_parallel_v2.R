local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
  library(mco)
  library(doParallel)
  library(foreach)
  library(tibble)
})

cat("Loading Distillation Targets 2.0...\n")
targets <- readRDS("distillation_targets_v2.rds")

# 1. Global Platt Scaling Calibration
cat("Calibrating the Teacher's Logits via Platt Scaling...\n")
fit <- glm(targets$labels ~ targets$rnn_logits, family = binomial)
calibrated_rnn_logits <- predict(fit, type = "link")
teacher_soft_targets <- plogis(calibrated_rnn_logits)
p_val <- teacher_soft_targets

# Extract elements
N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
lag_Reward <- targets$lag_reward
lag_Ch <- targets$lag_ch
lag_Resp <- targets$lag_resp

# The Exact Distillation Targets (Phase 2.0)
rnn_p_logits <- calibrated_rnn_logits
rnn_mu <- targets$rnn_mu
rnn_sigma <- targets$rnn_sigma

# Bio Initialization
set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

# NSGA-II High-Density Parameters
POPSIZE <- 256
GENERATIONS <- 100

cat("Starting Parallel Cluster on 32 Cores...\n")
cl <- makeCluster(32)
registerDoParallel(cl)

# =====================================================================
# CORTICO-CEREBELLAR DISTILLATION 2.0
# =====================================================================
cat("Starting Cortico-Cerebellar NSGA-II (Precision-Weighted)...\n")
cortico_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  
  results <- foreach(ind = 1:num_ind, .combine = cbind, .export=c("N_trials", "X", "W_gen", "W_ach1", "W_ach2", "W_thal", "Pi_vec", "ITI", "rnn_p_logits", "p_val", "rnn_mu", "rnn_sigma")) %dopar% {
    p_alpha_pc <- params_matrix[1, ind]
    p_lambda_pc <- params_matrix[2, ind]
    p_beta_thal <- params_matrix[3, ind]
    p_kappa_cf <- params_matrix[4, ind]
    p_alpha_gran <- params_matrix[5, ind]
    p_beta_gran <- params_matrix[6, ind]
    p_sigma2_diff <- params_matrix[7, ind]
    gamma <- params_matrix[8, ind]
    
    mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
    mu_history <- matrix(0, nrow=N_trials, ncol=32)
    diverged <- FALSE
    
    for (t in 1:N_trials) {
      if (ITI[t] > 0) {
        Z <- rep(0, 896)
        W_purk <- W_purk + rnorm(896, 0, sqrt(p_sigma2_diff * ITI[t]))
      }
      I_t <- X[t, ]
      I_hat <- as.numeric(W_gen %*% mu)
      eps <- Pi_vec * (I_t - I_hat)
      mu <- mu + p_alpha_pc * (as.numeric(t(W_gen) %*% eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * as.numeric(W_thal %*% D))
      
      if (any(is.na(mu)) || max(abs(mu)) > 1e4) { diverged <- TRUE; break }
      
      G <- as.numeric(W_ach1 %*% mu)
      Z <- (1.0 - p_beta_gran) * Z + p_alpha_gran * G
      eps_mag <- mean(abs(eps))
      W_purk <- W_purk - p_kappa_cf * (eps_mag * Z)
      D <- as.numeric(W_ach2 %*% (G * W_purk))
      mu_history[t, ] <- mu
    }
    
    if (diverged) {
      c(100.0, 1.0 - gamma)
    } else {
      XX_inv <- tryCatch(solve(crossprod(mu_history) + diag(0.01, 32)), error = function(e) NULL)
      if(is.null(XX_inv)) {
         c(100.0, 1.0 - gamma)
      } else {
        W_policy <- XX_inv %*% crossprod(mu_history, rnn_p_logits)
        W_mu <- XX_inv %*% crossprod(mu_history, rnn_mu)
        
        bio_logits <- as.numeric(mu_history %*% W_policy)
        bio_mu <- mu_history %*% W_mu
        
        blend_logits <- (1 - gamma) * rnn_p_logits + gamma * bio_logits
        blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu
        
        # Log-Normal Precision-Weighted MSE (Distributional matching)
        kinematic_loss <- mean( (rnn_mu - blend_mu)^2 / (2 * rnn_sigma^2) )
        
        blend_probs <- plogis(blend_logits)
        blend_probs <- pmax(pmin(blend_probs, 1 - 1e-7), 1e-7)
        bce <- -mean(p_val * log(blend_probs) + (1 - p_val) * log(1 - blend_probs))
        
        c(bce + kinematic_loss, 1.0 - gamma)
      }
    }
  }
  return(results)
}

res_bio_v2 <- nsga2(cortico_obj, idim = 8, odim = 2, 
                 lower.bounds = c(0.001, 0.001, 0.001, 0.0001, 0.001, 0.001, 0.001, 0.0), 
                 upper.bounds = c(1.500, 1.500, 0.500, 0.1000, 1.000, 1.000, 0.500, 1.0), 
                 popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)

saveRDS(res_bio_v2, "nsga2_moe_v2_results.rds")
cat("Cortico-Cerebellar Distillation 2.0 Done.\n")

stopCluster(cl)

# Create tracking tibble for analysis
df <- as_tibble(res_bio_v2$par, .name_repair = "unique")
colnames(df) <- c("alpha_pc", "kappa_cf", "lambda_pc", "beta_thal", "sigma2_diff", "sigma2_dec", "tau", "gamma")
df$Composite_NLL <- res_bio_v2$value[, 1]
df$Teacher_Penalty <- res_bio_v2$value[, 2]
df$Pareto_Optimal <- res_bio_v2$pareto.optimal
saveRDS(df, "moe_v2_pareto.rds")

cat("SUCCESS: Finished writing Pareto Front data to moe_v2_pareto.rds\n")
