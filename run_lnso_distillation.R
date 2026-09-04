library(torch)
library(dplyr)
library(jsonlite)
library(pROC)
library(PRROC)

stan_data <- read_json('data/stan_data_N30.json', simplifyVector = TRUE)
N_trials <- stan_data$N
N_subj <- stan_data$N_subj

Ch <- numeric(N_trials)
for(i in 1:N_trials) {
  Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])
}

Switch <- numeric(N_trials)
lag_Reward <- numeric(N_trials)
lag_Ch <- numeric(N_trials)

current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0 
    lag_Reward[i] <- 0
    lag_Ch[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]
    lag_Ch[i] <- Ch[i-1]
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

X <- cbind(X_features, lag_Reward, X_lag_Ch)
input_dim <- ncol(X)
hidden_dim <- 32

UniversalRNN <- nn_module(
  "UniversalRNN",
  initialize = function(input_dim, hidden_dim) {
    self$gru <- nn_gru(input_dim, hidden_dim, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$mu_head <- nn_linear(hidden_dim, 1)
    self$sigma_head <- nn_linear(hidden_dim, 1)
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    p_switch <- nnf_sigmoid(self$policy_head(h))
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(self$sigma_head(h)) + 1e-4
    list(p = p_switch, mu = mu_rt, sigma = sigma_rt, h = h)
  }
)

model <- UniversalRNN(input_dim, hidden_dim)
optimizer <- optim_adam(model$parameters, lr = 0.01)

set.seed(42)
val_subjs <- sample(1:N_subj, 6)
train_subjs <- setdiff(1:N_subj, val_subjs)

epochs <- 200
patience <- 10
best_val_loss <- Inf
epochs_no_improve <- 0
best_model_state <- NULL

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
      loss_policy <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
    } else {
      loss_policy <- torch_tensor(0)
    }
    
    log_rt <- torch_log(y_rt)
    dist <- distr_normal(preds$mu, preds$sigma)
    loss_rt <- -dist$log_prob(log_rt)$mean()
    loss <- loss_policy + loss_rt
    loss$backward()
    optimizer$step()
    train_loss <- train_loss + loss$item()
  }
  
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
      dist <- distr_normal(preds$mu, preds$sigma)
      loss_r <- -dist$log_prob(torch_log(y_rt))$mean()
      v_loss <- loss_p + loss_r
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
    exp_rt <- exp(as.numeric(preds$mu) + (as.numeric(preds$sigma)^2) / 2)
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

cat(sprintf("\n=== Decoupled LNSO Metrics ===\n"))
cat(sprintf("Global Val Loss: %.4f\n", best_val_loss))
cat(sprintf("Kinematic Precision (RT-RMSE): %.4f\n", rt_rmse))
cat(sprintf("Policy Discriminability (ROC-AUC): %.4f\n", roc_auc))
cat(sprintf("Policy Precision (PR-AUC): %.4f\n", pr_auc))
cat(sprintf("==============================\n"))

soft_p <- numeric()
soft_mu <- numeric()
soft_sigma <- numeric()
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
    soft_mu <- c(soft_mu, as.numeric(preds$mu))
    soft_sigma <- c(soft_sigma, as.numeric(preds$sigma))
    if (s == 1) { H <- as.matrix(preds$h[1, , ]) } else { H <- rbind(H, as.matrix(preds$h[1, , ])) }
  }
})

stan_data$soft_p <- soft_p
stan_data$soft_mu <- soft_mu
stan_data$soft_sigma <- soft_sigma
stan_data$H <- H

write_json(stan_data, 'data/stan_data_N30_distilled.json', auto_unbox = TRUE)
cat("LNSO Extraction complete.\n")
