library(torch)
library(dplyr)
library(jsonlite)
library(pROC)
library(PRROC)

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
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0
    current_subj <- stan_data_50$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data_50$Reward[i-1]
    lag_Ch[i] <- Ch[i-1]
    lag_RT[i] <- stan_data_50$RT[i-1]
  }
}

X_features <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_features[i, stan_data_50$Bd1[i]] <- 1; X_features[i, stan_data_50$Bd2[i]] <- 1 }
X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }

X <- cbind(X_features, lag_Reward, lag_RT, X_lag_Ch)
input_dim <- ncol(X); hidden_dim <- 32

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
epochs <- 40

all_preds_switch <- numeric(); all_true_switch <- numeric()
all_preds_rt <- numeric(); all_true_rt <- numeric()
all_S1 <- NULL; all_S2 <- NULL
mean_log_var_policy <- 0; mean_log_var_kin <- 0
M <- 1000

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
      
      loss_p <- if (sum(mask) > 0) nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE]) else torch_tensor(0)
      
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
      N_v <- length(idx)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
      preds <- model(x_t)
      
      mask <- c(FALSE, rep(TRUE, length(idx)-1))
      all_preds_switch <- c(all_preds_switch, as.numeric(preds$p)[mask])
      all_true_switch <- c(all_true_switch, Switch[idx][mask])
      
      pi_arr <- as.matrix(preds$pi[1, , ])
      mu_arr <- as.matrix(preds$mu[1, , ])
      sig_arr <- as.matrix(preds$sigma[1, , ])
      tau_arr <- as.matrix(preds$tau[1, , ])
      
      exp_rt_1 <- exp(mu_arr[,1] + (sig_arr[,1]^2)/2) + tau_arr[,1]
      exp_rt_2 <- exp(mu_arr[,2] + (sig_arr[,2]^2)/2) + tau_arr[,2]
      all_preds_rt <- c(all_preds_rt, pi_arr[,1] * exp_rt_1 + pi_arr[,2] * exp_rt_2)
      
      y_v <- stan_data_50$RT[idx]
      all_true_rt <- c(all_true_rt, y_v)
      
      # CRPS samples
      comp1_S1 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
      comp1_S2 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
      
      mu1_M <- rep(mu_arr[,1], M); sig1_M <- rep(sig_arr[,1], M); tau1_M <- rep(tau_arr[,1], M)
      mu2_M <- rep(mu_arr[,2], M); sig2_M <- rep(sig_arr[,2], M); tau2_M <- rep(tau_arr[,2], M)
      
      samp_S1 <- ifelse(comp1_S1 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
      samp_S2 <- ifelse(comp1_S2 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
      
      mat_S1 <- matrix(samp_S1, nrow = N_v, ncol = M, byrow = FALSE)
      mat_S2 <- matrix(samp_S2, nrow = N_v, ncol = M, byrow = FALSE)
      
      if (is.null(all_S1)) { all_S1 <- mat_S1; all_S2 <- mat_S2 } else { all_S1 <- rbind(all_S1, mat_S1); all_S2 <- rbind(all_S2, mat_S2) }
    }
  })
}

rt_rmse <- sqrt(mean((all_preds_rt - all_true_rt)^2))
E_Xy <- rowMeans(abs(all_S1 - all_true_rt))
E_XX <- rowMeans(abs(all_S1 - all_S2))
crps_per_trial <- E_Xy - 0.5 * E_XX
mean_crps <- mean(crps_per_trial)

roc_obj <- roc(all_true_switch, all_preds_switch, direction="<", quiet=TRUE)
pr_obj <- pr.curve(scores.class0 = all_preds_switch[all_true_switch == 1], scores.class1 = all_preds_switch[all_true_switch == 0], curve = FALSE)

cat(sprintf("\\n=== MDN (K=2, L=2) LNSO Metrics (N=50) ===\\n"))
cat(sprintf("Kinematic Precision (RT-RMSE): %.4f\\n", rt_rmse))
cat(sprintf("Kinematic Precision (CRPS): %.4f\\n", mean_crps))
cat(sprintf("Policy Discriminability (ROC-AUC): %.4f\\n", as.numeric(auc(roc_obj))))
cat(sprintf("Policy Precision (PR-AUC): %.4f\\n", pr_obj$auc.integral))
cat(sprintf("Average Learned Log-Var Policy: %.4f\\n", mean_log_var_policy))
cat(sprintf("Average Learned Log-Var Kinematic: %.4f\\n", mean_log_var_kin))
cat(sprintf("===========================================\n"))
