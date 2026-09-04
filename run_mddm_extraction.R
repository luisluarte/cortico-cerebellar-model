library(torch)
library(dplyr)
library(jsonlite)

stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
N_trials <- stan_data$N
N_subj <- stan_data$N_subj
min_RT <- min(stan_data$RT)

Ch <- numeric(N_trials)
for(i in 1:N_trials) {
  Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])
}

Switch <- numeric(N_trials)
lag_Reward <- numeric(N_trials)
lag_Ch <- numeric(N_trials)
lag_RT <- numeric(N_trials)

current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0 
    lag_Reward[i] <- 0
    lag_Ch[i] <- 0
    lag_RT[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]
    lag_Ch[i] <- Ch[i-1]
    lag_RT[i] <- stan_data$RT[i-1]
  }
}

X_features <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
  X_features[i, stan_data$Bd1[i]] <- 1
  X_features[i, stan_data$Bd2[i]] <- 1
}

X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
  if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1
}

X <- cbind(X_features, lag_Reward, lag_RT, X_lag_Ch)
input_dim <- ncol(X)
hidden_dim <- 32

UniversalRNN <- nn_module(
  "UniversalRNN",
  initialize = function(input_dim, hidden_dim) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    
    self$mu_head <- nn_linear(hidden_dim, 2)
    self$sigma_head <- nn_linear(hidden_dim, 2)
    self$tau_head <- nn_linear(hidden_dim, 2)
    self$pi_head <- nn_linear(hidden_dim, 2)
    
    self$log_var_policy <- nn_parameter(torch_zeros(1))
    self$log_var_kin <- nn_parameter(torch_zeros(1))
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    p_switch <- nnf_sigmoid(self$policy_head(h))
    
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(self$sigma_head(h)) + 1e-4
    tau_rt <- nnf_sigmoid(self$tau_head(h)) * (0.99 * min_RT)
    pi_mix <- nnf_softmax(self$pi_head(h), dim = -1)
    
    list(p = p_switch, mu = mu_rt, sigma = sigma_rt, tau = tau_rt, pi = pi_mix, h = h)
  }
)

model <- UniversalRNN(input_dim, hidden_dim)
optimizer <- optim_adam(model$parameters, lr = 0.01)

set.seed(42)
val_subjs <- sample(1:N_subj, 20)
train_subjs <- setdiff(1:N_subj, val_subjs)

epochs <- 200
patience <- 10
best_val_loss <- Inf
epochs_no_improve <- 0
best_model_state <- NULL

for (ep in 1:epochs) {
  model$train()
  for (s in train_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
    
    optimizer$zero_grad()
    preds <- model(x_t)
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    
    if (sum(mask) > 0) {
      loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
    } else {
      loss_p <- torch_tensor(0)
    }
    
    y_rt_shifted <- y_rt - preds$tau
    log_y <- torch_log(y_rt_shifted)
    
    log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
    log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
    loss_r <- -log_mix$mean()
    
    total_loss <- torch_exp(-model$log_var_policy) * loss_p + model$log_var_policy + 
                  torch_exp(-model$log_var_kin) * loss_r + model$log_var_kin
                  
    total_loss$backward()
    optimizer$step()
  }
  
  model$eval()
  val_loss <- 0
  with_no_grad({
    for (s in val_subjs) {
      idx <- which(stan_data$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
      y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
      preds <- model(x_t)
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      
      if (sum(mask) > 0) {
         loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
      } else {
         loss_p <- torch_tensor(0)
      }
      
      y_rt_shifted <- y_rt - preds$tau
      log_y <- torch_log(y_rt_shifted)
      log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      
      v_loss <- torch_exp(-model$log_var_policy) * loss_p + model$log_var_policy + 
                torch_exp(-model$log_var_kin) * loss_r + model$log_var_kin
                
      val_loss <- val_loss + v_loss$item()
    }
  })
  val_loss <- val_loss / length(val_subjs)
  
  if (val_loss < best_val_loss) {
    best_val_loss <- val_loss
    epochs_no_improve <- 0
    best_model_state <- lapply(model$parameters, function(x) x$clone())
  } else {
    epochs_no_improve <- epochs_no_improve + 1
  }
  
  if (epochs_no_improve >= patience) {
    break
  }
}

with_no_grad({
  for (name in names(best_model_state)) {
    model$parameters[[name]]$copy_(best_model_state[[name]])
  }
})

model$eval()

# Extraction
soft_p <- numeric()
soft_pi_1 <- numeric(); soft_mu_1 <- numeric(); soft_sigma_1 <- numeric(); soft_tau_1 <- numeric()
soft_pi_2 <- numeric(); soft_mu_2 <- numeric(); soft_sigma_2 <- numeric(); soft_tau_2 <- numeric()
H <- NULL

with_no_grad({
  for (s in 1:N_subj) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    preds <- model(x_t)
    
    p_sw <- as.numeric(preds$p)
    p_c1 <- numeric(length(p_sw))
    p_c1[1] <- 0.5
    for (t_idx in 2:length(idx)) {
      prev_ch <- lag_Ch[idx[t_idx]]
      if (stan_data$Bd1[idx[t_idx]] == prev_ch) {
        p_c1[t_idx] <- 1 - p_sw[t_idx]
      } else if (stan_data$Bd2[idx[t_idx]] == prev_ch) {
        p_c1[t_idx] <- p_sw[t_idx]
      } else {
        p_c1[t_idx] <- 0.5
      }
    }
    soft_p <- c(soft_p, p_c1)
    
    pi_arr <- as.matrix(preds$pi[1, , ])
    mu_arr <- as.matrix(preds$mu[1, , ])
    sig_arr <- as.matrix(preds$sigma[1, , ])
    tau_arr <- as.matrix(preds$tau[1, , ])
    
    E1 <- exp(mu_arr[,1] + (sig_arr[,1]^2)/2) + tau_arr[,1]
    E2 <- exp(mu_arr[,2] + (sig_arr[,2]^2)/2) + tau_arr[,2]
    
    is_1_fast <- E1 <= E2
    
    soft_pi_1 <- c(soft_pi_1, ifelse(is_1_fast, pi_arr[,1], pi_arr[,2]))
    soft_mu_1 <- c(soft_mu_1, ifelse(is_1_fast, mu_arr[,1], mu_arr[,2]))
    soft_sigma_1 <- c(soft_sigma_1, ifelse(is_1_fast, sig_arr[,1], sig_arr[,2]))
    soft_tau_1 <- c(soft_tau_1, ifelse(is_1_fast, tau_arr[,1], tau_arr[,2]))
    
    soft_pi_2 <- c(soft_pi_2, ifelse(is_1_fast, pi_arr[,2], pi_arr[,1]))
    soft_mu_2 <- c(soft_mu_2, ifelse(is_1_fast, mu_arr[,2], mu_arr[,1]))
    soft_sigma_2 <- c(soft_sigma_2, ifelse(is_1_fast, sig_arr[,2], sig_arr[,1]))
    soft_tau_2 <- c(soft_tau_2, ifelse(is_1_fast, tau_arr[,2], tau_arr[,1]))
    
    if (s == 1) { H <- as.matrix(preds$h[1, , ]) } else { H <- rbind(H, as.matrix(preds$h[1, , ])) }
  }
})

stan_data$soft_p <- soft_p
stan_data$soft_pi_1 <- soft_pi_1
stan_data$soft_mu_1 <- soft_mu_1
stan_data$soft_sigma_1 <- soft_sigma_1
stan_data$soft_tau_1 <- soft_tau_1
stan_data$soft_pi_2 <- soft_pi_2
stan_data$soft_mu_2 <- soft_mu_2
stan_data$soft_sigma_2 <- soft_sigma_2
stan_data$soft_tau_2 <- soft_tau_2
stan_data$H <- H

stan_data$soft_expected_rt <- NULL
stan_data$soft_var_rt <- NULL
stan_data$soft_mu <- NULL
stan_data$soft_sigma <- NULL
stan_data$soft_tau <- NULL

write_json(stan_data, 'data/stan_data_N100_mddm.json', auto_unbox = TRUE)
cat("Sorted mDDM Extraction complete.\n")
