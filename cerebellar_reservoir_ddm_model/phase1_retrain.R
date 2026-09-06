library(torch)
library(jsonlite)
library(pROC)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
N_trials <- stan_data$N
min_RT <- min(stan_data$RT)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

Ch <- numeric(N_trials)
for(i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials); lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
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

for (p in model_final$parameters) {
  p$requires_grad_(FALSE)
}
torch_save(model_final$state_dict(), "frozen_rnn_baseline.pt")
cat("Final model frozen and saved to 'frozen_rnn_baseline.pt'. Phase 1 Complete!\n")
