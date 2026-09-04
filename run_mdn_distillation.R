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
    
    # MDN Heads (K=2)
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

cat("Training N100 MDN (K=2) LNSO Distillation...\n")

for (ep in 1:epochs) {
  model$train()
  train_loss <- 0
  for (s in train_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    # y_rt shape: (1, T, 1) -> broadcast to (1, T, 2)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
    
    optimizer$zero_grad()
    preds <- model(x_t)
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    
    if (sum(mask) > 0) {
      loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
    } else {
      loss_p <- torch_tensor(0)
    }
    
    # MDN Lognormal Log-Likelihood
    # log_p_k = -log(y - tau) - 0.5*log(2pi) - log(sigma) - 0.5 * ((log(y - tau) - mu) / sigma)^2
    y_rt_shifted <- y_rt - preds$tau
    log_y <- torch_log(y_rt_shifted)
    
    log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
    log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
    loss_r <- -log_mix$mean()
    
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
val_preds_switch <- numeric()
val_true_switch <- numeric()
val_preds_rt <- numeric()
val_true_rt <- numeric()

with_no_grad({
  for (s in val_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    preds <- model(x_t)
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    val_preds_switch <- c(val_preds_switch, as.numeric(preds$p)[mask])
    val_true_switch <- c(val_true_switch, Switch[idx][mask])
    
    # MDN Expected RT = pi_1*(exp(mu_1 + sigma_1^2 / 2) + tau_1) + pi_2*(exp(mu_2 + sigma_2^2 / 2) + tau_2)
    pi_arr <- as.matrix(preds$pi[1, , ])
    mu_arr <- as.matrix(preds$mu[1, , ])
    sig_arr <- as.matrix(preds$sigma[1, , ])
    tau_arr <- as.matrix(preds$tau[1, , ])
    
    exp_rt_1 <- exp(mu_arr[,1] + (sig_arr[,1]^2)/2) + tau_arr[,1]
    exp_rt_2 <- exp(mu_arr[,2] + (sig_arr[,2]^2)/2) + tau_arr[,2]
    exp_rt <- pi_arr[,1] * exp_rt_1 + pi_arr[,2] * exp_rt_2
    
    val_preds_rt <- c(val_preds_rt, exp_rt)
    val_true_rt <- c(val_true_rt, stan_data$RT[idx])
  }
})

rt_rmse <- sqrt(mean((val_preds_rt - val_true_rt)^2))
roc_obj <- roc(val_true_switch, val_preds_switch, direction="<", quiet=TRUE)
roc_auc <- as.numeric(auc(roc_obj))
pr_obj <- pr.curve(scores.class0 = val_preds_switch[val_true_switch == 1],
                   scores.class1 = val_preds_switch[val_true_switch == 0],
                   curve = FALSE)
pr_auc <- pr_obj$auc.integral

cat(sprintf("\n=== MDN (K=2) LNSO Metrics (N=100) ===\n"))
cat(sprintf("Global Val Loss: %.4f\n", best_val_loss))
cat(sprintf("Kinematic Precision (RT-RMSE): %.4f\n", rt_rmse))
cat(sprintf("Policy Discriminability (ROC-AUC): %.4f\n", roc_auc))
cat(sprintf("Policy Precision (PR-AUC): %.4f\n", pr_auc))
cat(sprintf("Learned Log-Var Policy: %.4f\n", as.numeric(model$log_var_policy)))
cat(sprintf("Learned Log-Var Kinematic: %.4f\n", as.numeric(model$log_var_kin)))
cat(sprintf("======================================\n"))
