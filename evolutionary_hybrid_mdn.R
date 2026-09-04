library(torch)
library(mco)
library(jsonlite)

stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
min_RT <- min(stan_data$RT)
subjs_50 <- readRDS('excluded_subjects.rds')
idx_50 <- which(stan_data$subj %in% subjs_50)
stan_data_50 <- list(
  subj = stan_data$subj[idx_50],
  Bd1 = stan_data$Bd1[idx_50],
  Bd2 = stan_data$Bd2[idx_50],
  Resp = stan_data$Resp[idx_50],
  Reward = stan_data$Reward[idx_50],
  RT = stan_data$RT[idx_50]
)
N_trials <- length(idx_50)

Ch <- numeric(N_trials)
for(i in 1:N_trials) {
  Ch[i] <- ifelse(stan_data_50$Resp[i] == 1, stan_data_50$Bd1[i], stan_data_50$Bd2[i])
}

Switch <- numeric(N_trials)
lag_Reward <- numeric(N_trials)
lag_Ch <- numeric(N_trials)
lag_RT <- numeric(N_trials)

current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data_50$subj[i] != current_subj) {
    Switch[i] <- 0 
    lag_Reward[i] <- 0
    lag_Ch[i] <- 0
    lag_RT[i] <- 0
    current_subj <- stan_data_50$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data_50$Reward[i-1]
    lag_Ch[i] <- Ch[i-1]
    lag_RT[i] <- stan_data_50$RT[i-1]
  }
}

X_features <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
  X_features[i, stan_data_50$Bd1[i]] <- 1
  X_features[i, stan_data_50$Bd2[i]] <- 1
}

X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
  if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1
}

X <- cbind(X_features, lag_Reward, lag_RT, X_lag_Ch)
input_dim <- ncol(X)
hidden_dim <- 32

L_pure <- readRDS('L_pure_mdn.rds')
K <- 5
set.seed(42)
folds <- sample(rep(1:K, length.out = length(subjs_50)))
test_subjs <- subjs_50[folds == 1]
train_subjs <- subjs_50[folds != 1]
baseline_L_pure <- mean(L_pure[as.character(test_subjs)])

HybridUniversalRNN <- nn_module(
  "HybridUniversalRNN",
  initialize = function(input_dim, hidden_dim, N_ratio, p) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 2, batch_first = TRUE)
    
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$mu_head <- nn_linear(hidden_dim, 2)
    self$sigma_head <- nn_linear(hidden_dim, 2)
    self$tau_head <- nn_linear(hidden_dim, 2)
    self$pi_head <- nn_linear(hidden_dim, 2)
    
    N_granule <- as.integer(round(hidden_dim * N_ratio))
    # Correct Achlioptas scaling by 1/sqrt(d)
    elements <- sample(c(-sqrt(3)/sqrt(hidden_dim), 0, sqrt(3)/sqrt(hidden_dim)), N_granule * hidden_dim, replace=TRUE, prob=c(p/2, 1-p, p/2))
    W_mat <- matrix(elements, nrow=hidden_dim, ncol=N_granule)
    self$W_res <- torch_tensor(W_mat, dtype=torch_float(), requires_grad=FALSE)
    
    self$pc_policy_head <- nn_linear(N_granule, 1)
    self$pc_mu_head <- nn_linear(N_granule, 2)
    self$pc_sigma_head <- nn_linear(N_granule, 2)
    self$pc_tau_head <- nn_linear(N_granule, 2)
    self$pc_pi_head <- nn_linear(N_granule, 2)
    
    self$log_var_policy <- nn_parameter(torch_zeros(1))
    self$log_var_kin <- nn_parameter(torch_zeros(1))
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    
    z <- torch_matmul(h$detach(), self$W_res)
    
    # Clip extreme logits before sigmoid/softmax to avoid NaNs
    logits_p <- torch_clamp(self$policy_head(h) + self$pc_policy_head(z), min = -15, max = 15)
    p_switch <- nnf_sigmoid(logits_p)
    
    mu_rt <- self$mu_head(h) + self$pc_mu_head(z)
    
    logits_sigma <- torch_clamp(self$sigma_head(h) + self$pc_sigma_head(z), min = -15, max = 15)
    sigma_rt <- nnf_softplus(logits_sigma) + 1e-4
    
    logits_tau <- torch_clamp(self$tau_head(h) + self$pc_tau_head(z), min = -15, max = 15)
    tau_rt <- nnf_sigmoid(logits_tau) * (0.99 * min_RT)
    
    logits_pi <- torch_clamp(self$pi_head(h) + self$pc_pi_head(z), min = -15, max = 15)
    pi_mix <- nnf_softmax(logits_pi, dim = -1)
    
    list(p = p_switch, mu = mu_rt, sigma = sigma_rt, tau = tau_rt, pi = pi_mix, h = h)
  }
)

evaluate_genome <- function(params) {
  N_ratio <- params[1]
  p <- params[2]
  
  model <- HybridUniversalRNN(input_dim, hidden_dim, N_ratio, p)
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  
  for (ep in 1:15) { 
    model$train()
    for (s in train_subjs) {
      idx <- which(stan_data_50$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
      y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data_50$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
      
      optimizer$zero_grad()
      preds <- model(x_t)
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      
      if (sum(mask) > 0) {
        # use nnf_binary_cross_entropy_with_logits if possible, but preds$p is already sigmoid
        loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
      } else {
        loss_p <- torch_tensor(0)
      }
      
      y_rt_shifted <- y_rt - preds$tau
      log_y <- torch_log(y_rt_shifted + 1e-6) # safe log
      log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      
      total_loss <- torch_exp(-torch_clamp(model$log_var_policy, min=-5, max=5)) * loss_p + model$log_var_policy + 
                    torch_exp(-torch_clamp(model$log_var_kin, min=-5, max=5)) * loss_r + model$log_var_kin
      
      total_loss$backward()
      
      # Clip gradients
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      
      optimizer$step()
    }
  }
  
  L_hybrid_vec <- numeric(length(test_subjs))
  model$eval()
  with_no_grad({
    for (i in seq_along(test_subjs)) {
      s <- test_subjs[i]
      idx <- which(stan_data_50$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
      y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data_50$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
      
      preds <- model(x_t)
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      
      if (sum(mask) > 0) {
        loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
      } else {
        loss_p <- torch_tensor(0)
      }
      
      y_rt_shifted <- y_rt - preds$tau
      log_y <- torch_log(y_rt_shifted + 1e-6)
      log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      
      v_loss <- torch_exp(-torch_clamp(model$log_var_policy, min=-5, max=5)) * loss_p + model$log_var_policy + 
                torch_exp(-torch_clamp(model$log_var_kin, min=-5, max=5)) * loss_r + model$log_var_kin
                
      L_hybrid_vec[i] <- v_loss$item()
    }
  })
  
  # if NaN, penalize
  L_hybrid <- mean(L_hybrid_vec)
  if (is.nan(L_hybrid) || is.na(L_hybrid)) {
    return(c(1e6, 0)) # Terrible fitness
  }
  
  delta_L <- abs(L_hybrid - baseline_L_pure)
  
  w_rnn <- c(
    as.numeric(model$policy_head$weight),
    as.numeric(model$mu_head$weight),
    as.numeric(model$sigma_head$weight),
    as.numeric(model$tau_head$weight),
    as.numeric(model$pi_head$weight)
  )
  w_pc <- c(
    as.numeric(model$pc_policy_head$weight),
    as.numeric(model$pc_mu_head$weight),
    as.numeric(model$pc_sigma_head$weight),
    as.numeric(model$pc_tau_head$weight),
    as.numeric(model$pc_pi_head$weight)
  )
  
  norm_rnn <- sum(w_rnn^2)
  norm_pc <- sum(w_pc^2)
  reliance <- norm_pc / (norm_pc + norm_rnn + 1e-8)
  
  cat(sprintf("[Eval] N_ratio=%.1f, p=%.3f -> Delta L=%.4f (L_hybrid=%.4f), Offloading=%.2f%%\\n", N_ratio, p, delta_L, L_hybrid, reliance * 100))
  return(c(delta_L, -reliance))
}

cat("Starting MDN NSGA-II Evolution (Pop: 20, Gen: 10)...\\n")
res <- nsga2(evaluate_genome, idim = 2, odim = 2, 
             lower.bounds = c(1, 0.01), upper.bounds = c(20, 0.5), 
             popsize = 20, generations = 10)

saveRDS(res, 'nsga2_mdn_results.rds')
cat("Evolution complete! Saved to nsga2_mdn_results.rds\\n")
