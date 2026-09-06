library(torch)
library(jsonlite)
library(pROC)
library(mco)

cat("=== Phase 4: OOB Evaluation of Top 5 Pareto Models ===\n")

# 1. Load Data & Subjects
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

# Ensure we use the 50 OUT OF BAG participants
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
if(length(scaffold_subjs) != 50) {
  cat(sprintf("Warning: Found %d OOB subjects, expected 50.\n", length(scaffold_subjs)))
}

mask <- stan_data$subj %in% scaffold_subjs
stan_data$subj <- stan_data$subj[mask]
stan_data$Resp <- stan_data$Resp[mask]
stan_data$Bd1 <- stan_data$Bd1[mask]
stan_data$Bd2 <- stan_data$Bd2[mask]
stan_data$Reward <- stan_data$Reward[mask]
stan_data$RT <- stan_data$RT[mask]
N_trials <- sum(mask)
min_RT <- min(stan_data$RT)

Ch <- numeric(N_trials)
for(i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])
Switch <- numeric(N_trials); lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials); ITI <- numeric(N_trials)
current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0; ITI[i] <- 5.0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
    ITI[i] <- 0.0
  }
}
X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)
Y_switch <- Switch
Y_RT <- stan_data$RT

# 2. Extract Top 5 Pareto Models
res <- readRDS("nsga2_moe_results.rds")
vals <- res$value
pars <- res$par
sort_idx <- order(vals[, 1])
top5_pars <- pars[sort_idx[1:5], ]

# 3. Load Frozen RNN Ceiling & Generate Fixed Reservoir
cat("Loading Frozen Ceilings and generating structural reservoirs...\n")
SpatialRNN <- nn_module(
  "SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$pi_head <- nn_linear(hidden_dim, K)
    self$mu_head <- nn_linear(hidden_dim, K)
    self$sigma_head <- nn_linear(hidden_dim, K)
    self$tau_head <- nn_linear(hidden_dim, K)
    self$log_var_policy <- nn_parameter(torch_zeros(1))
    self$log_var_kin <- nn_parameter(torch_zeros(1))
    self$K <- K
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
    p_switch <- nnf_sigmoid(logits_p)
    pi_mix <- nnf_softmax(torch_clamp(self$pi_head(h), min=-15, max=15), dim=-1)
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(torch_clamp(self$sigma_head(h), min=-15, max=15)) + 1e-4
    tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min=-15, max=15)) * (0.99 * min_RT)
    list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

x_t <- torch_tensor(X, dtype = torch_float())$unsqueeze(1)
with_no_grad({
  preds <- model_rnn(x_t)
  p_val <- as.matrix(preds$p$squeeze())
  p_val[p_val < 1e-7] <- 1e-7
  p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
  rnn_mu <- as.matrix(preds$mu$squeeze())
  rnn_sigma <- as.matrix(preds$sigma$squeeze())
  rnn_tau <- as.matrix(preds$tau$squeeze())
  rnn_pi <- as.matrix(preds$pi$squeeze())
})

set.seed(42) # MUST match NSGA-II initialization precisely!
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

# Helper Metrics
calculate_pr_auc <- function(probs, labels) {
  ord <- order(probs, decreasing = TRUE)
  probs <- probs[ord]; labels <- labels[ord]
  tp <- cumsum(labels == 1); fp <- cumsum(labels == 0)
  precision <- tp / (tp + fp); recall <- tp / sum(labels == 1)
  recall_diff <- c(recall[1], diff(recall))
  sum(recall_diff * precision, na.rm = TRUE)
}
crps_normal <- function(y, mu, sigma) {
  z <- (y - mu) / sigma
  sigma * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi))
}

cat(sprintf("\n%-10s | %-10s | %-10s | %-10s | %-10s | %-30s\n", "Model", "ROC-AUC", "PR-AUC", "MCC", "CRPS", "Confusion Matrix (TP,TN,FP,FN)"))
cat(rep("-", 90), "\n", sep="")

# 4. Evaluation Loop
for (m in 1:5) {
  theta <- top5_pars[m, ]
  alpha_pc      <- theta[1]; lambda_pc     <- theta[2]; beta_thal     <- theta[3]; kappa_cf      <- theta[4]
  alpha_granule <- theta[5]; beta_granule  <- theta[6]; sigma2_diff   <- theta[7]; gamma_raw     <- theta[8]
  
  # Run Euler once for all 50 OOB subjects
  mu_history <- matrix(0, nrow=N_trials, ncol=32)
  mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
  
  for (t in 1:N_trials) {
    if (ITI[t] > 0) {
      Z[] <- 0; W_purk <- W_purk + rnorm(896, 0, sqrt(sigma2_diff * ITI[t]))
    }
    I_t <- X[t, ]
    I_hat <- as.numeric(W_gen %*% mu)
    eps <- Pi_vec * (I_t - I_hat)
    mu <- mu + alpha_pc * (as.numeric(t(W_gen) %*% eps) - (lambda_pc * ITI[t]) * mu + beta_thal * as.numeric(W_thal %*% D))
    G <- as.numeric(W_ach1 %*% mu)
    Z <- (1 - beta_granule) * Z + alpha_granule * G
    eps_mag <- mean(abs(eps))
    W_purk <- W_purk - kappa_cf * (eps_mag * Z)
    D <- as.numeric(W_ach2 %*% (G * W_purk))
    mu_history[t, ] <- mu
  }
  
  # 10-Fold LNSO Validation
  folds <- split(scaffold_subjs, ceiling(seq_along(scaffold_subjs)/5)) # 10 folds of 5 subjects
  metrics <- list(roc=c(), pr=c(), mcc=c(), crps=c(), tp=c(), tn=c(), fp=c(), fn=c())
  
  for (f in 1:length(folds)) {
    test_subjs <- folds[[f]]
    train_idx <- which(!(stan_data$subj %in% test_subjs))
    test_idx <- which(stan_data$subj %in% test_subjs)
    
    # Train Ridge
    mu_train <- mu_history[train_idx, , drop=FALSE]
    lambda_ridge <- 0.01
    XX_inv <- solve(crossprod(mu_train) + diag(lambda_ridge, 32))
    W_policy <- XX_inv %*% crossprod(mu_train, rnn_p_logits[train_idx])
    W_mu <- XX_inv %*% crossprod(mu_train, rnn_mu[train_idx, , drop=FALSE])
    
    # Test Bio Projection
    mu_test <- mu_history[test_idx, , drop=FALSE]
    bio_policy_logits <- mu_test %*% W_policy
    bio_mu <- mu_test %*% W_mu
    
    # Blend MoE
    gamma <- 1 / (1 + exp(-gamma_raw))
    blend_policy_logits <- (1 - gamma) * rnn_p_logits[test_idx] + gamma * bio_policy_logits
    blend_p <- 1 / (1 + exp(-blend_policy_logits))
    blend_mu <- (1 - gamma) * rnn_mu[test_idx, , drop=FALSE] + gamma * bio_mu
    
    # Calculate Metrics
    y_true <- Y_switch[test_idx]
    roc <- as.numeric(pROC::auc(y_true, blend_p, quiet=TRUE))
    pr <- calculate_pr_auc(blend_p, y_true)
    
    preds <- ifelse(blend_p > 0.5, 1, 0)
    TP <- sum(preds == 1 & y_true == 1); TN <- sum(preds == 0 & y_true == 0)
    FP <- sum(preds == 1 & y_true == 0); FN <- sum(preds == 0 & y_true == 1)
    N <- length(y_true)
    
    num <- (TP * TN) - (FP * FN)
    den <- sqrt(as.numeric(TP+FP) * as.numeric(TP+FN) * as.numeric(TN+FP) * as.numeric(TN+FN))
    mcc <- if (den == 0) 0 else num / den
    
    # CRPS (Merged Normal Approx)
    pi_test <- rnn_pi[test_idx, , drop=FALSE]
    sigma_test <- rnn_sigma[test_idx, , drop=FALSE]
    mu_m <- pi_test[,1]*blend_mu[,1] + pi_test[,2]*blend_mu[,2]
    var_m <- pi_test[,1]*(sigma_test[,1]^2 + blend_mu[,1]^2) + pi_test[,2]*(sigma_test[,2]^2 + blend_mu[,2]^2) - mu_m^2
    sigma_m <- sqrt(pmax(var_m, 1e-6))
    crps <- mean(crps_normal(Y_RT[test_idx], mu_m, sigma_m))
    
    metrics$roc <- c(metrics$roc, roc); metrics$pr <- c(metrics$pr, pr)
    metrics$mcc <- c(metrics$mcc, mcc); metrics$crps <- c(metrics$crps, crps)
    metrics$tp <- c(metrics$tp, TP/N); metrics$tn <- c(metrics$tn, TN/N)
    metrics$fp <- c(metrics$fp, FP/N); metrics$fn <- c(metrics$fn, FN/N)
  }
  
  cat(sprintf("Model %-4d | %.4f+-%.3f | %.4f+-%.3f | %.4f+-%.3f | %.4f+-%.3f | TP:%.3f(%.3f), TN:%.3f(%.3f), FP:%.3f(%.3f), FN:%.3f(%.3f)\n",
      m, 
      mean(metrics$roc), sd(metrics$roc),
      mean(metrics$pr), sd(metrics$pr),
      mean(metrics$mcc), sd(metrics$mcc),
      mean(metrics$crps), sd(metrics$crps),
      mean(metrics$tp), sd(metrics$tp),
      mean(metrics$tn), sd(metrics$tn),
      mean(metrics$fp), sd(metrics$fp),
      mean(metrics$fn), sd(metrics$fn)
      ))
}
cat("=== OOB Evaluation Complete ===\n")
