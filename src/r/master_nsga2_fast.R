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
p_val <- teacher_soft_targets

# Extract elements
N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
lag_Reward <- targets$lag_reward
lag_Ch <- targets$lag_ch
lag_Resp <- targets$lag_resp

# The Exact Distillation Targets
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

POPSIZE <- 128
GENERATIONS <- 40

cat("Starting Parallel Cluster on 32 Cores...\n")
cl <- makeCluster(32)
registerDoParallel(cl)

cat("Compiling C++ Bio Simulator on all 32 Cores...\n")
clusterEvalQ(cl, {
  library(Rcpp)
  library(RcppArmadillo)
  sourceCpp("sim_bio.cpp")
})

# =====================================================================
# 1. WSLS (Distillation 2.0)
# =====================================================================
cat("Starting WSLS NSGA-II...\n")
wsls_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  
  results <- foreach(ind = 1:num_ind, .combine = cbind, .export=c("lag_Reward", "lag_Ch", "rnn_p_logits", "p_val", "rnn_mu", "rnn_sigma")) %dopar% {
    theta_win <- max(1e-7, min(1-1e-7, params_matrix[1, ind]))
    theta_loss <- max(1e-7, min(1-1e-7, params_matrix[2, ind]))
    gamma <- params_matrix[3, ind]
    
    wsls_p <- ifelse(lag_Reward > 0, theta_win, theta_loss)
    wsls_p[lag_Reward == 0 & lag_Ch == 0] <- 0.5 
    
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * qlogis(wsls_p)
    blend_probs <- pmax(pmin(plogis(blend_logits), 1 - 1e-7), 1e-7)
    bce <- -mean(p_val * log(blend_probs) + (1 - p_val) * log(1 - blend_probs))
    
    State <- matrix(lag_Reward, ncol=1)
    XX_inv <- solve(crossprod(State) + diag(0.01, 1))
    W_mu <- XX_inv %*% crossprod(State, rnn_mu)
    wsls_mu <- State %*% W_mu
    blend_mu <- (1 - gamma) * rnn_mu + gamma * wsls_mu
    
    # Precision-Weighted Kinematic Loss
    kinematic_loss <- mean((rnn_mu - blend_mu)^2 / (2 * rnn_sigma^2))
    
    c(bce + kinematic_loss, 1.0 - gamma)
  }
  return(results)
}
res_wsls <- nsga2(wsls_obj, idim = 3, odim = 2, lower.bounds = c(0.001, 0.001, 0.0), upper.bounds = c(0.999, 0.999, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
saveRDS(res_wsls, "fast_res_wsls.rds")
cat("WSLS Done.\n")


# =====================================================================
# 2. Q-LEARNING (Distillation 2.0)
# =====================================================================
cat("Starting Q-Learning NSGA-II...\n")
q_learning_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  
  results <- foreach(ind = 1:num_ind, .combine = cbind, .export=c("N_trials", "lag_Resp", "lag_Ch", "rnn_p_logits", "p_val", "rnn_mu", "rnn_sigma", "targets")) %dopar% {
    alpha_win <- params_matrix[1, ind]
    alpha_loss <- params_matrix[2, ind]
    beta <- params_matrix[3, ind]
    gamma <- params_matrix[4, ind]
    
    Q <- rep(0.5, 8)
    q_logits <- numeric(N_trials)
    Q_history <- matrix(0, nrow=N_trials, ncol=8)
    
    stan_data_subj <- targets$subjs
    stan_data_bd1 <- max.col(targets$X[, 1:8])
    stan_data_bd2 <- max.col(targets$X[, 9:16])
    stan_data_resp <- targets$lag_resp
    stan_data_reward <- targets$lag_reward
    
    current_subj <- -1
    for (t in 1:N_trials) {
      if (stan_data_subj[t] != current_subj) { Q <- rep(0.5, 8); current_subj <- stan_data_subj[t] }
      Q_history[t, ] <- Q
      b1 <- stan_data_bd1[t]; b2 <- stan_data_bd2[t]
      p_bd1 <- plogis(beta * (Q[b1] - Q[b2]))
      if (lag_Resp[t] == 1) {
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
      } else {
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
      }
      p_switch <- max(1e-7, min(1-1e-7, p_switch))
      q_logits[t] <- qlogis(p_switch)
      
      chosen <- ifelse(stan_data_resp[t] == 1, b1, b2)
      unchosen <- ifelse(stan_data_resp[t] == 1, b2, b1)
      r <- stan_data_reward[t]
      if (r > 0) {
         Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen])
         Q[unchosen] <- Q[unchosen] + alpha_win * (0 - Q[unchosen])
      } else {
         Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen])
         Q[unchosen] <- Q[unchosen] + alpha_loss * (1 - Q[unchosen])
      }
    }
    
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * q_logits
    blend_probs <- pmax(pmin(plogis(blend_logits), 1 - 1e-7), 1e-7)
    bce <- -mean(p_val * log(blend_probs) + (1 - p_val) * log(1 - blend_probs))
    
    XX_inv <- tryCatch(solve(crossprod(Q_history) + diag(0.01, 8)), error = function(e) NULL)
    if(is.null(XX_inv)) { 
      rmse_mu <- 10.0 
      kinematic_loss <- 10.0
    } else {
      W_mu <- XX_inv %*% crossprod(Q_history, rnn_mu)
      q_mu <- Q_history %*% W_mu
      blend_mu <- (1 - gamma) * rnn_mu + gamma * q_mu
      kinematic_loss <- mean((rnn_mu - blend_mu)^2 / (2 * rnn_sigma^2))
    }
    c(bce + kinematic_loss, 1.0 - gamma)
  }
  return(results)
}
res_qlearn <- nsga2(q_learning_obj, idim = 4, odim = 2, lower.bounds = c(0.001, 0.001, 0.1, 0.0), upper.bounds = c(0.999, 0.999, 10.0, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
saveRDS(res_qlearn, "fast_res_qlearn.rds")
cat("Q-Learning Done.\n")


# =====================================================================
# 3. CORTICO-CEREBELLAR (Distillation 2.0 with Rcpp)
# =====================================================================
cat("Starting Cortico-Cerebellar NSGA-II (Precision-Weighted via Rcpp)...\n")
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
    
    # Call the lightning-fast C++ function
    mu_history <- simulate_bio(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_vec, p_alpha_pc, p_lambda_pc, p_beta_thal, p_kappa_cf, p_alpha_gran, p_beta_gran, p_sigma2_diff)
    
    if (nrow(mu_history) == 1 && is.na(mu_history[1,1])) {
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
saveRDS(res_bio_v2, "fast_res_bio.rds")
cat("Cortico-Cerebellar Done.\n")

stopCluster(cl)

# Compile the fast results into a single dataset for plotting!
tib <- tibble(
  Model = c(rep("WSLS", nrow(res_wsls$value)), rep("QLearn", nrow(res_qlearn$value)), rep("Bio", nrow(res_bio_v2$value))),
  NLL = c(res_wsls$value[,1], res_qlearn$value[,1], res_bio_v2$value[,1]),
  Teacher_Penalty = c(res_wsls$value[,2], res_qlearn$value[,2], res_bio_v2$value[,2]),
  Gamma = 1 - c(res_wsls$value[,2], res_qlearn$value[,2], res_bio_v2$value[,2])
)

saveRDS(tib, "distillation_2_fast_pareto.rds")
cat("ALL DONE! Saved to distillation_2_fast_pareto.rds\n")
