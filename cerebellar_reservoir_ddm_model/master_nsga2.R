library(mco)
library(jsonlite)
library(torch)
library(doParallel)
library(tibble)

cat("Loading Data and isolating the 50 Knowledge Distillation subjects...\n")
stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
# Use the RNN training set for Knowledge Distillation to leave the other 50 subjects 100% untouched for LNSO
mask <- stan_data$subj %in% rnn_phase1_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)

Switch <- numeric(N_trials)
Ch <- ifelse(stan_data$Resp == 1, stan_data$Bd1, stan_data$Bd2)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
  }
}
ITI <- c(5.0, rep(0.0, N_trials - 1))
current_subj <- -1
for(i in 1:N_trials) {
    if (stan_data$subj[i] != current_subj) { ITI[i] <- 5.0; current_subj <- stan_data$subj[i] }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

cat("Loading RNN Oracle...\n")
SpatialRNN <- nn_module("SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$pi_head <- nn_linear(hidden_dim, K)
    self$mu_head <- nn_linear(hidden_dim, K)
    self$sigma_head <- nn_linear(hidden_dim, K)
    self$tau_head <- nn_linear(hidden_dim, K)
  },
  forward = function(x, h0 = NULL) {
    h <- self$gru(x, h0)[[1]]
    list(p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)), mu = self$mu_head(h))
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

with_no_grad({
  preds <- model_rnn(torch_tensor(X, dtype=torch_float())$unsqueeze(1))
  p_val <- as.numeric(preds$p$squeeze())
  p_val[p_val < 1e-7] <- 1e-7; p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
  rnn_mu <- as.matrix(preds$mu$squeeze())
})

# Initialize random projection matrices for Bio Model
set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

POPSIZE <- 200
GENERATIONS <- 150

cat("Starting Parallel Cluster...\n")
cl <- makeCluster(32)
registerDoParallel(cl)

# =====================================================================
# 1. PARAMETRIZED WSLS
# =====================================================================
cat("Starting WSLS NSGA-II...\n")
wsls_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  results <- matrix(0, nrow=2, ncol=num_ind)
  State <- matrix(lag_Reward, ncol=1)
  
  for (ind in 1:num_ind) {
    theta_win <- max(1e-7, min(1-1e-7, params_matrix[1, ind]))
    theta_loss <- max(1e-7, min(1-1e-7, params_matrix[2, ind]))
    gamma <- params_matrix[3, ind]
    
    wsls_p <- ifelse(lag_Reward > 0, theta_win, theta_loss)
    wsls_p[lag_Reward == 0 & lag_Ch == 0] <- 0.5 
    
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * qlogis(wsls_p)
    bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
    
    XX_inv <- solve(crossprod(State) + diag(0.01, 1))
    W_mu <- XX_inv %*% crossprod(State, rnn_mu)
    wsls_mu <- State %*% W_mu
    blend_mu <- (1 - gamma) * rnn_mu + gamma * wsls_mu
    rmse_mu <- sqrt(mean((rnn_mu - blend_mu)^2))
    
    results[1, ind] <- bce + rmse_mu
    results[2, ind] <- 1.0 - gamma
  }
  return(results)
}
clusterExport(cl, c("wsls_obj", "stan_data", "N_trials", "lag_Reward", "lag_Ch", "rnn_p_logits", "p_val", "rnn_mu"))

res_wsls <- nsga2(wsls_obj, idim = 3, odim = 2, lower.bounds = c(0.001, 0.001, 0.0), upper.bounds = c(0.999, 0.999, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
saveRDS(res_wsls, "master_res_wsls.rds")
cat("WSLS Done.\n")

# =====================================================================
# 2. Q-LEARNING
# =====================================================================
cat("Starting Q-Learning NSGA-II...\n")
q_learning_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  results <- matrix(0, nrow=2, ncol=num_ind)
  
  for (ind in 1:num_ind) {
    alpha_win <- params_matrix[1, ind]
    alpha_loss <- params_matrix[2, ind]
    beta <- params_matrix[3, ind]
    gamma <- params_matrix[4, ind]
    
    Q <- rep(0.5, 8)
    q_logits <- numeric(N_trials)
    Q_history <- matrix(0, nrow=N_trials, ncol=8)
    
    current_subj <- -1
    for (t in 1:N_trials) {
      if (stan_data$subj[t] != current_subj) { Q <- rep(0.5, 8); current_subj <- stan_data$subj[t] }
      Q_history[t, ] <- Q
      b1 <- stan_data$Bd1[t]; b2 <- stan_data$Bd2[t]
      p_bd1 <- plogis(beta * (Q[b1] - Q[b2]))
      if (lag_Resp[t] == 1) {
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
      } else {
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1 else p_switch <- p_bd1
      }
      p_switch <- max(1e-7, min(1-1e-7, p_switch))
      q_logits[t] <- qlogis(p_switch)
      
      chosen <- ifelse(stan_data$Resp[t] == 1, b1, b2)
      unchosen <- ifelse(stan_data$Resp[t] == 1, b2, b1)
      r <- stan_data$Reward[t]
      if (r > 0) {
         Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen])
         Q[unchosen] <- Q[unchosen] + alpha_win * (0 - Q[unchosen])
      } else {
         Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen])
         Q[unchosen] <- Q[unchosen] + alpha_loss * (1 - Q[unchosen])
      }
    }
    
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * q_logits
    bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
    
    XX_inv <- tryCatch(solve(crossprod(Q_history) + diag(0.01, 8)), error = function(e) NULL)
    if(is.null(XX_inv)) { rmse_mu <- 10.0 } else {
      W_mu <- XX_inv %*% crossprod(Q_history, rnn_mu)
      q_mu <- Q_history %*% W_mu
      blend_mu <- (1 - gamma) * rnn_mu + gamma * q_mu
      rmse_mu <- sqrt(mean((rnn_mu - blend_mu)^2))
    }
    results[1, ind] <- bce + rmse_mu
    results[2, ind] <- 1.0 - gamma
  }
  return(results)
}
clusterExport(cl, c("q_learning_obj", "stan_data", "N_trials", "lag_Resp", "lag_Ch", "rnn_p_logits", "p_val", "rnn_mu"))

res_qlearn <- nsga2(q_learning_obj, idim = 4, odim = 2, lower.bounds = c(0.001, 0.001, 0.1, 0.0), upper.bounds = c(0.999, 0.999, 10.0, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
saveRDS(res_qlearn, "master_res_qlearn.rds")
cat("Q-Learning Done.\n")

# =====================================================================
# 3. CORTICO-CEREBELLAR
# =====================================================================
cat("Starting Cortico-Cerebellar NSGA-II...\n")
cortico_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
  num_ind <- ncol(params_matrix)
  results <- matrix(0, nrow=2, ncol=num_ind)
  
  for (ind in 1:num_ind) {
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
      results[1, ind] <- 100.0
      results[2, ind] <- 1.0 - gamma
      next
    }
    
    XX_inv <- tryCatch(solve(crossprod(mu_history) + diag(0.01, 32)), error = function(e) NULL)
    if(is.null(XX_inv)) {
       results[1, ind] <- 100.0
       results[2, ind] <- 1.0 - gamma
       next
    }
    
    W_policy <- XX_inv %*% crossprod(mu_history, rnn_p_logits)
    W_mu <- XX_inv %*% crossprod(mu_history, rnn_mu)
    
    bio_logits <- mu_history %*% W_policy
    bio_mu <- mu_history %*% W_mu
    
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * bio_logits
    blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu
    
    bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
    rmse_mu <- sqrt(mean((rnn_mu - blend_mu)^2))
    
    results[1, ind] <- bce + rmse_mu
    results[2, ind] <- 1.0 - gamma
  }
  return(results)
}
clusterExport(cl, c("cortico_obj", "stan_data", "N_trials", "X", "W_gen", "W_ach1", "W_ach2", "W_thal", "Pi_vec", "ITI", "rnn_p_logits", "p_val", "rnn_mu"))

res_bio <- nsga2(cortico_obj, idim = 8, odim = 2, 
                 lower.bounds = c(0.001, 0.001, 0.001, 0.0001, 0.001, 0.001, 0.001, 0.0), 
                 upper.bounds = c(1.500, 1.500, 0.500, 0.1000, 1.000, 1.000, 0.500, 1.0), 
                 popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
saveRDS(res_bio, "master_res_bio.rds")
cat("Cortico-Cerebellar Done.\n")

stopCluster(cl)

# Combine into Tibble
tib <- tibble(
  Model = c(rep("WSLS", nrow(res_wsls$value)), rep("QLearn", nrow(res_qlearn$value)), rep("Bio", nrow(res_bio$value))),
  NLL = c(res_wsls$value[,1], res_qlearn$value[,1], res_bio$value[,1]),
  RNN_Reliance = c(res_wsls$value[,2], res_qlearn$value[,2], res_bio$value[,2]),
  Gamma = 1 - c(res_wsls$value[,2], res_qlearn$value[,2], res_bio$value[,2]),
  Params = c(lapply(1:nrow(res_wsls$par), function(i) res_wsls$par[i,]), 
             lapply(1:nrow(res_qlearn$par), function(i) res_qlearn$par[i,]), 
             lapply(1:nrow(res_bio$par), function(i) res_bio$par[i,]))
)
saveRDS(tib, "master_pareto_fronts.rds")
cat("ALL DONE. Saved to master_pareto_fronts.rds\n")
