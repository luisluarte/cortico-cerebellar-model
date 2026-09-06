library(torch)
library(jsonlite)
library(pROC)

calculate_pr_auc <- function(probs, labels) {
  ord <- order(probs, decreasing = TRUE)
  probs <- probs[ord]
  labels <- labels[ord]
  tp <- cumsum(labels == 1)
  fp <- cumsum(labels == 0)
  precision <- tp / (tp + fp)
  recall <- tp / sum(labels == 1)
  recall_diff <- c(recall[1], diff(recall))
  return(sum(recall_diff * precision, na.rm = TRUE))
}

# Ensure CUDA is available
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("Running on device: %s\n", as.character(device)))

# 1. Load Data
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
N_trials <- stan_data$N
min_RT <- min(stan_data$RT)

# 2. Extract 50 unique subjects for RNN Phase 1 
all_subjs <- unique(stan_data$subj)
set.seed(42)
rnn_phase1_subjs <- sample(all_subjs, 50)
saveRDS(rnn_phase1_subjs, "rnn_phase1_subjects.rds")
cat("Isolated 50 subjects and saved to rnn_phase1_subjects.rds\n")

# Process features
Ch <- numeric(N_trials)
for(i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials); lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials)
lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
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

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# 3. Model Definition
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

# 4. LNSO Setup (5 folds of 10 subjects out)
folds <- split(sample(rnn_phase1_subjs), rep(1:5, each=10))
cv_auc <- numeric(5)
cv_pr_auc <- numeric(5)
cv_crps <- numeric(5)
cv_acc <- numeric(5)

cat("\n=== Starting 5-Fold LNSO Cross-Validation ===\n")
for (f in 1:5) {
  cat(sprintf("\n--- Fold %d ---\n", f))
  test_subjs <- folds[[f]]
  train_subjs <- setdiff(rnn_phase1_subjs, test_subjs)
  
  torch_manual_seed(42 + f)
  model <- SpatialRNN(ncol(X), 4, 2)
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = 0.002, weight_decay = 0.0)
  
  # Training Loop
  for (ep in 1:20) {
    model$train()
    for (s in train_subjs) {
      idx <- which(stan_data$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float(), device = device)$unsqueeze(1)
      y_switch <- torch_tensor(Switch[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
      
      optimizer$zero_grad()
      preds <- model(x_t)
      
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      if (sum(mask) > 0) {
        p_t <- preds$p[, mask, , drop=FALSE]
        y_t <- y_switch[, mask, , drop=FALSE]
        n_switch <- y_t$sum()$item()
        n_stay <- y_t$numel() - n_switch
        pos_weight <- if (n_switch > 0) n_stay / n_switch else 1.0
        bce <- -(y_t * torch_log(p_t + 1e-7) * torch_tensor(pos_weight, device = device) + (1 - y_t) * torch_log(1 - p_t + 1e-7))
        loss_p <- bce$mean()
      } else {
        loss_p <- torch_tensor(0, device = device)
      }
      
      log_y <- torch_log(y_rt - preds$tau + 1e-6)
      log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      
      total_loss <- torch_exp(-torch_clamp(model$log_var_policy, min=-5, max=5)) * loss_p + model$log_var_policy + torch_exp(-torch_clamp(model$log_var_kin, min=-5, max=5)) * loss_r + model$log_var_kin
      
      total_loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()
    }
  }
  
  # Evaluation Loop
  model$eval()
  all_preds_switch <- numeric()
  all_true_switch <- numeric()
  all_true_rt <- numeric()
  all_S1 <- NULL; all_S2 <- NULL
  M <- 200
  
  with_no_grad({
    for (s in test_subjs) {
      idx <- which(stan_data$subj == s)
      N_v <- length(idx)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float(), device = device)$unsqueeze(1)
      preds <- model(x_t)
      
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      all_preds_switch <- c(all_preds_switch, as.numeric(preds$p$cpu())[mask])
      all_true_switch <- c(all_true_switch, Switch[idx][mask])
      all_true_rt <- c(all_true_rt, stan_data$RT[idx])
      
      pi_arr <- as.matrix(preds$pi[1, , ]$cpu())
      mu_arr <- as.matrix(preds$mu[1, , ]$cpu())
      sig_arr <- as.matrix(preds$sigma[1, , ]$cpu())
      tau_arr <- as.matrix(preds$tau[1, , ]$cpu())
      
      comp1_S1 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
      comp1_S2 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
      
      mu1_M <- rep(mu_arr[,1], M); sig1_M <- rep(sig_arr[,1], M); tau1_M <- rep(tau_arr[,1], M)
      mu2_M <- rep(mu_arr[,2], M); sig2_M <- rep(sig_arr[,2], M); tau2_M <- rep(tau_arr[,2], M)
      
      samp_S1 <- ifelse(comp1_S1 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
      samp_S2 <- ifelse(comp1_S2 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
      
      mat_S1 <- matrix(samp_S1, nrow = N_v, ncol = M, byrow = FALSE)
      mat_S2 <- matrix(samp_S2, nrow = N_v, ncol = M, byrow = FALSE)
      
      if (is.null(all_S1)) {
        all_S1 <- mat_S1; all_S2 <- mat_S2
      } else {
        all_S1 <- rbind(all_S1, mat_S1); all_S2 <- rbind(all_S2, mat_S2)
      }
    }
  })
  
  roc_obj <- roc(all_true_switch, all_preds_switch, direction="<", quiet=TRUE)
  cv_auc[f] <- as.numeric(auc(roc_obj))
  cv_pr_auc[f] <- calculate_pr_auc(all_preds_switch, all_true_switch)
  preds_binary <- ifelse(all_preds_switch > 0.5, 1, 0)
  cm <- table(Prediction = preds_binary, Reference = all_true_switch)
  cv_acc[f] <- sum(diag(cm))/sum(cm)
  cv_crps[f] <- mean(rowMeans(abs(all_S1 - all_true_rt)) - 0.5 * rowMeans(abs(all_S1 - all_S2)))
  
  cat(sprintf("Fold %d Results -> ROC-AUC: %.4f, PR-AUC: %.4f, CRPS: %.4f, Accuracy: %.4f\n", f, cv_auc[f], cv_pr_auc[f], cv_crps[f], cv_acc[f]))
}

cat("\n=== Final LNSO Cross-Validation Metrics ===\n")
cat(sprintf("Mean ROC-AUC: %.4f (SD: %.4f)\n", mean(cv_auc), sd(cv_auc)))
cat(sprintf("Mean PR-AUC:  %.4f (SD: %.4f)\n", mean(cv_pr_auc), sd(cv_pr_auc)))
cat(sprintf("Mean CRPS:    %.4f (SD: %.4f)\n", mean(cv_crps), sd(cv_crps)))
cat(sprintf("Mean Acc:     %.4f (SD: %.4f)\n", mean(cv_acc), sd(cv_acc)))

# 5. Retrain Final Frozen Model on all 50 Subjects
cat("\nRetraining final frozen model on all 50 isolated subjects...\n")
model_final <- SpatialRNN(ncol(X), 4, 2)
model_final <- model_final$to(device = device)
optimizer <- optim_adam(model_final$parameters, lr = 0.002, weight_decay = 0.0)

for (ep in 1:20) {
  model_final$train()
  for (s in rnn_phase1_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float(), device = device)$unsqueeze(1)
    y_switch <- torch_tensor(Switch[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
    
    optimizer$zero_grad()
    preds <- model_final(x_t)
    
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    if (sum(mask) > 0) {
      p_t <- preds$p[, mask, , drop=FALSE]
      y_t <- y_switch[, mask, , drop=FALSE]
      n_switch <- y_t$sum()$item()
      n_stay <- y_t$numel() - n_switch
      pos_weight <- if (n_switch > 0) n_stay / n_switch else 1.0
      bce <- -(y_t * torch_log(p_t + 1e-7) * torch_tensor(pos_weight, device = device) + (1 - y_t) * torch_log(1 - p_t + 1e-7))
      loss_p <- bce$mean()
    } else {
      loss_p <- torch_tensor(0, device = device)
    }
    
    log_y <- torch_log(y_rt - preds$tau + 1e-6)
    log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
    log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
    loss_r <- -log_mix$mean()
    
    total_loss <- torch_exp(-torch_clamp(model_final$log_var_policy, min=-5, max=5)) * loss_p + model_final$log_var_policy + torch_exp(-torch_clamp(model_final$log_var_kin, min=-5, max=5)) * loss_r + model_final$log_var_kin
    
    total_loss$backward()
    nn_utils_clip_grad_norm_(model_final$parameters, max_norm = 1.0)
    optimizer$step()
  }
}

model_final$requires_grad_(FALSE)
torch_save(model_final, "frozen_rnn_baseline.pt")
cat("Final model frozen and saved to 'frozen_rnn_baseline.pt'. Phase 1 Complete!\n")
