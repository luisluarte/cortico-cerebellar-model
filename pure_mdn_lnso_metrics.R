library(torch)
library(dplyr)
library(jsonlite)
library(pROC)
library(PRROC)

cat("Loading Data...\\n")
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

UniversalRNN <- nn_module(
  "UniversalRNN",
  initialize = function(input_dim, hidden_dim) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 2, batch_first = TRUE)
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

K <- 5
set.seed(42)
folds <- sample(rep(1:K, length.out = length(subjs_50)))

L_pure <- numeric(length(subjs_50))
names(L_pure) <- subjs_50
epochs <- 40 # 40 epochs for reasonable convergence per fold

all_preds_switch <- numeric()
all_true_switch <- numeric()
all_preds_rt <- numeric()
all_true_rt <- numeric()
mean_log_var_policy <- 0
mean_log_var_kin <- 0

cat("Starting Full 5-Fold LNSO Evaluation...\\n")

for (k in 1:K) {
  test_subjs <- subjs_50[folds == k]
  train_subjs <- subjs_50[folds != k]
  
  model <- UniversalRNN(input_dim, hidden_dim)
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  
  for (ep in 1:epochs) {
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
        loss_p <- nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])
      } else {
        loss_p <- torch_tensor(0)
      }
      
      y_rt_shifted <- y_rt - preds$tau
      log_y <- torch_log(y_rt_shifted + 1e-6)
      log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      
      total_loss <- torch_exp(-torch_clamp(model$log_var_policy, min=-5, max=5)) * loss_p + model$log_var_policy + 
                    torch_exp(-torch_clamp(model$log_var_kin, min=-5, max=5)) * loss_r + model$log_var_kin
      
      total_loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()
    }
  }
  
  mean_log_var_policy <- mean_log_var_policy + as.numeric(model$log_var_policy) / K
  mean_log_var_kin <- mean_log_var_kin + as.numeric(model$log_var_kin) / K
  
  model$eval()
  with_no_grad({
    for (s in test_subjs) {
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
                
      L_pure[as.character(s)] <- v_loss$item()
      
      # Predictions for Metrics
      all_preds_switch <- c(all_preds_switch, as.numeric(preds$p)[mask])
      all_true_switch <- c(all_true_switch, Switch[idx][mask])
      
      pi_arr <- as.matrix(preds$pi[1, , ])
      mu_arr <- as.matrix(preds$mu[1, , ])
      sig_arr <- as.matrix(preds$sigma[1, , ])
      tau_arr <- as.matrix(preds$tau[1, , ])
      
      exp_rt_1 <- exp(mu_arr[,1] + (sig_arr[,1]^2)/2) + tau_arr[,1]
      exp_rt_2 <- exp(mu_arr[,2] + (sig_arr[,2]^2)/2) + tau_arr[,2]
      exp_rt <- pi_arr[,1] * exp_rt_1 + pi_arr[,2] * exp_rt_2
      
      all_preds_rt <- c(all_preds_rt, exp_rt)
      all_true_rt <- c(all_true_rt, stan_data_50$RT[idx])
    }
  })
  cat(sprintf("Fold %d complete.\\n", k))
}

saveRDS(L_pure, 'L_pure_mdn.rds')

rt_rmse <- sqrt(mean((all_preds_rt - all_true_rt)^2))
roc_obj <- roc(all_true_switch, all_preds_switch, direction="<", quiet=TRUE)
roc_auc <- as.numeric(auc(roc_obj))
pr_obj <- pr.curve(scores.class0 = all_preds_switch[all_true_switch == 1],
                   scores.class1 = all_preds_switch[all_true_switch == 0],
                   curve = FALSE)
pr_auc <- pr_obj$auc.integral

cat(sprintf("\\n=== MDN (K=2, L=2) LNSO Metrics (N=50) ===\\n"))
cat(sprintf("Global Val Loss (Cross-Entropy NLL): %.4f\\n", mean(L_pure)))
cat(sprintf("Kinematic Precision (RT-RMSE): %.4f\\n", rt_rmse))
cat(sprintf("Policy Discriminability (ROC-AUC): %.4f\\n", roc_auc))
cat(sprintf("Policy Precision (PR-AUC): %.4f\\n", pr_auc))
cat(sprintf("Average Learned Log-Var Policy: %.4f\\n", mean_log_var_policy))
cat(sprintf("Average Learned Log-Var Kinematic: %.4f\\n", mean_log_var_kin))
cat(sprintf("===========================================\n"))
