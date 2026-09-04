library(torch)
library(dplyr)
library(jsonlite)
library(pROC)
library(PRROC)

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
    self$gru <- nn_gru(input_dim, hidden_dim, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$mu_head <- nn_linear(hidden_dim, 1)
    self$sigma_head <- nn_linear(hidden_dim, 1)
    self$tau_head <- nn_linear(hidden_dim, 1)
    
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
    list(p = p_switch, mu = mu_rt, sigma = sigma_rt, tau = tau_rt, h = h)
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

cat("Training Single-Component Lognormal with CRPS...\n")

for (ep in 1:epochs) {
  model$train()
  train_loss <- 0
  for (s in train_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    
    optimizer$zero_grad()
    preds <- model(x_t)
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    
    if (sum(mask) > 0) {
      loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
    } else {
      loss_p <- torch_tensor(0)
    }
    
    y_rt_shifted <- y_rt - preds$tau
    log_rt <- torch_log(y_rt_shifted)
    dist <- distr_normal(preds$mu, preds$sigma)
    loss_r <- -dist$log_prob(log_rt)$mean()
    
    total_loss <- torch_exp(-model$log_var_policy) * loss_p + model$log_var_policy + 
                  torch_exp(-model$log_var_kin) * loss_r + model$log_var_kin
                  
    total_loss$backward()
    optimizer$step()
    train_loss <- train_loss + total_loss$item()
  }
  train_loss <- train_loss / length(train_subjs)
  
  model$eval()
  val_loss <- 0
  with_no_grad({
    for (s in val_subjs) {
      idx <- which(stan_data$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
      y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
      preds <- model(x_t)
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      
      if (sum(mask) > 0) {
         loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
      } else {
         loss_p <- torch_tensor(0)
      }
      y_rt_shifted <- y_rt - preds$tau
      dist <- distr_normal(preds$mu, preds$sigma)
      loss_r <- -dist$log_prob(torch_log(y_rt_shifted))$mean()
      
      v_loss <- torch_exp(-model$log_var_policy) * loss_p + model$log_var_policy + 
                torch_exp(-model$log_var_kin) * loss_r + model$log_var_kin
                
      val_loss <- val_loss + v_loss$item()
    }
  })
  val_loss <- val_loss / length(val_subjs)
  
  cat(sprintf("Epoch %d | Train Loss: %.4f | Val Loss: %.4f\n", ep, train_loss, val_loss))
  
  if (val_loss < best_val_loss) {
    best_val_loss <- val_loss
    epochs_no_improve <- 0
    best_model_state <- lapply(model$parameters, function(x) x$clone())
  } else {
    epochs_no_improve <- epochs_no_improve + 1
  }
  
  if (epochs_no_improve >= patience) {
    cat(sprintf("Early stopping at epoch %d. Best Val Loss: %.4f\n", ep, best_val_loss))
    break
  }
}

with_no_grad({
  for (name in names(best_model_state)) {
    model$parameters[[name]]$copy_(best_model_state[[name]])
  }
})

model$eval()
val_true_rt <- numeric()
M <- 1000
all_S1 <- NULL
all_S2 <- NULL

with_no_grad({
  for (s in val_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    preds <- model(x_t)
    
    mu_arr <- as.numeric(preds$mu)
    sig_arr <- as.numeric(preds$sigma)
    tau_arr <- as.numeric(preds$tau)
    
    N_v <- length(idx)
    y_v <- stan_data$RT[idx]
    val_true_rt <- c(val_true_rt, y_v)
    
    mu_M <- rep(mu_arr, M)
    sig_M <- rep(sig_arr, M)
    tau_M <- rep(tau_arr, M)
    
    samp_S1 <- rlnorm(N_v * M, mu_M, sig_M) + tau_M
    samp_S2 <- rlnorm(N_v * M, mu_M, sig_M) + tau_M
    
    mat_S1 <- matrix(samp_S1, nrow = N_v, ncol = M, byrow = FALSE)
    mat_S2 <- matrix(samp_S2, nrow = N_v, ncol = M, byrow = FALSE)
    
    if (is.null(all_S1)) {
      all_S1 <- mat_S1
      all_S2 <- mat_S2
    } else {
      all_S1 <- rbind(all_S1, mat_S1)
      all_S2 <- rbind(all_S2, mat_S2)
    }
  }
})

E_Xy <- rowMeans(abs(all_S1 - val_true_rt))
E_XX <- rowMeans(abs(all_S1 - all_S2))
crps_per_trial <- E_Xy - 0.5 * E_XX
mean_crps <- mean(crps_per_trial)

cat(sprintf("\n=== Single Lognormal CRPS Metric ===\n"))
cat(sprintf("Kinematic Precision (CRPS): %.4f\n", mean_crps))
cat(sprintf("====================================\n"))
