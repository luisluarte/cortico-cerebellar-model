# setup ----

# set local lib path
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# set download preferences
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://cloud.r-project.org"
  options(repos = r)
})

# install package manager
cat("\n############ installing pacman #############\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman", lib = local_lib)
}

cat("\n###### loading libs #########\n")
pacman::p_load(
  tidyverse,
  torch,
  jsonlite,
  pROC,
  cli
)

cli::cli_h2("setting workspace dir")
if (dir.exists("src/r")) {
  setwd("src/r")
}
cli::cli_text(getwd())

cli::cli_h1("starting script execution")

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

cli::cli_h2("checking for CUDA availability")
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cli::cli_text(sprintf("running on device: %s\n", as.character(device)))

cli::cli_h2("participant selection for phase 1")
stan_data <- read_json("../../data/stan_data.json", simplifyVector = TRUE)
N_trials <- stan_data$N
min_RT <- min(stan_data$RT)
all_subjs <- unique(stan_data$subj)
set.seed(42)
rnn_phase1_subjs <- sample(all_subjs, 50)
cli::cli_alert_info("randomly selected 50 participants for RNN phase 1")
print(rnn_phase1_subjs)
saveRDS(rnn_phase1_subjs, "rnn_phase1_subjects.rds")
cli::cli_alert_info(
  "Selected participants save at {file.path(getwd(), 'rnn_phase1_subjects.rds')}"
)

cli::cli_alert_info("generating features...")

# continous rewards
min_bd <- min(c(stan_data$Bd1, stan_data$Bd2))
max_bd <- max(c(stan_data$Bd1, stan_data$Bd2))

# min max normalization of continous rewards
X_Bd1 <- (stan_data$Bd1 - min_bd) / (max_bd - min_bd)
X_Bd2 <- (stan_data$Bd2 - min_bd) / (max_bd - min_bd)

# continous value
Ch_cont <- ifelse(stan_data$Resp == 1, X_Bd1, X_Bd2)

# Binary choice & lagged features ----
df_lags <- tibble(
  subj    = stan_data$subj,
  Resp    = stan_data$Resp,
  Reward  = stan_data$Reward,
  RT      = stan_data$RT,
  Ch_cont = Ch_cont
) %>%
  group_by(subj) %>%
  mutate(
    Switch          = coalesce(as.numeric(Resp != lag(Resp)), 0),
    lag_Reward_bin  = lag(Reward, default = 0),
    lag_Reward_cont = lag(Ch_cont, default = 0),
    lag_RT          = lag(RT, default = 0),
    lag_Resp        = if_else(lag(Resp, default = 0) == 1, 1, 0)
  ) %>%
  ungroup()

# extract vectors
Switch <- df_lags$Switch
lag_Reward_bin <- df_lags$lag_Reward_bin
lag_Reward_cont <- df_lags$lag_Reward_cont
lag_RT <- df_lags$lag_RT
lag_Resp <- df_lags$lag_Resp


# 6d state vector
X <- cbind(X_Bd1, X_Bd2, lag_Reward_bin, lag_Reward_cont, lag_Resp, lag_RT)

# model definition ----
cli::cli_alert_info("defining teacher model architecture")
# this is the teacher network, it predicts two behavioral outputs at the same time
# (1) probability of switchin and (2) reaction time
# for (1) is just p(switch=1 | state)
# for (2) RT_{t} is shifted log-normal mixture
SpatialRNN <- nn_module(
  "SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    # a gated recurrent unit (GRU) provide short-term memory to the model
    # update gate = how much of past memory to retain, get h^{t-1} AND X^{t} over a sigmoid
    # this outputs z^{t} which is the update gate value
    # reset gate = how much of past memory to forget, get h^{t-1} AND X^{t} over a sigmoid
    # this outputs r^{t} which is the reset gate value
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    # policy head is the linear readout of the policy networks for the bernoulli
    # switch decision
    self$policy_head <- nn_linear(hidden_dim, 1)
    # pi_head is the linear readout mapping the hidden state to the mixing proportions
    # over the K components of the Gaussian mixture model for reaction times
    self$pi_head <- nn_linear(hidden_dim, K)
    # mu_head is the linear readout mapping the hidden state to the means
    # of the Gaussian mixture model for reaction times
    self$mu_head <- nn_linear(hidden_dim, K)
    # sigma_head is the linear readout mapping the hidden state to the standard deviations
    # of the Gaussian mixture model for reaction times
    self$sigma_head <- nn_linear(hidden_dim, K)
    # tau_head is the linear readout mapping the hidden state to the mixture weights
    # of the Gaussian mixture model for reaction times
    self$tau_head <- nn_linear(hidden_dim, K)
    # log_var_policy is the log variance of the bernoulli distribution for switch decision
    # log_var_kin is the log variance of the Gaussian mixture model for reaction times
    self$log_var_policy <- nn_parameter(torch_zeros(1))
    self$log_var_kin <- nn_parameter(torch_zeros(1))
    self$K <- K
  },
  # execute the trial-by-trial pass through the network
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
    p_switch <- nnf_sigmoid(logits_p)
    pi_mix <- nnf_softmax(torch_clamp(self$pi_head(h), min = -15, max = 15), dim = -1)
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(torch_clamp(self$sigma_head(h), min = -15, max = 15)) + 1e-4
    tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min = -15, max = 15)) * (0.99 * min_RT)
    list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
  }
)

cli::cli_alert_info("setting up for leave-one-subject-out cross-validation")
# lnso setup (5 folds of 10 subjects out)
folds <- split(sample(rnn_phase1_subjs), rep(1:5, each = 10))
cv_auc <- numeric(5)
cv_pr_auc <- numeric(5)
cv_crps <- numeric(5)
cv_acc <- numeric(5)

cli::cli_alert_info("starting 5-fold LNSO cross validation")
for (f in 1:5) {
  cli::cli_alert_info(sprintf("Fold %d", f))
  test_subjs <- folds[[f]]
  train_subjs <- setdiff(rnn_phase1_subjs, test_subjs)

  torch_manual_seed(42 + f)
  model <- SpatialRNN(ncol(X), 4, 2)
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = 0.002, weight_decay = 0.0)

  # training loop
  for (ep in 1:20) {
    model$train()
    for (s in train_subjs) {
      idx <- which(stan_data$subj == s)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float(), device = device)$unsqueeze(1)
      y_switch_tensor <- torch_tensor(Switch[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)
      y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float(), device = device)$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
      optimizer$zero_grad()
      preds <- model(x_t)

      mask <- c(FALSE, rep(TRUE, length(idx) - 1))
      if (sum(mask) > 0) {
        p_t <- preds$p[, mask, , drop = FALSE]
        y_t <- y_switch_tensor[, mask, , drop = FALSE]
        n_switch <- y_t$sum()$item()
        n_stay <- y_t$numel() - n_switch
        pos_weight <- if (n_switch > 0) n_stay / n_switch else 1.0
        bce <- -(y_t * torch_log(p_t + 1e-7) * torch_tensor(pos_weight, device = device) + (1 - y_t) * torch_log(1 - p_t + 1e-7))
        loss_p <- bce$mean()
      } else {
        loss_p <- torch_tensor(0, device = device)
      }
      log_y <- torch_log(y_rt - preds$tau + 1e-6)
      log_p_k <- -log_y - 0.5 * log(2 * pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
      log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
      loss_r <- -log_mix$mean()
      total_loss <- torch_exp(-torch_clamp(model$log_var_policy, min = -5, max = 5)) * loss_p + model$log_var_policy + torch_exp(-torch_clamp(model$log_var_kin, min = -5, max = 5)) * loss_r + model$log_var_kin
      total_loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()
    }
  }

  model$eval()
  all_preds_switch <- numeric()
  all_true_switch <- numeric()
  all_true_rt <- numeric()
  all_S1 <- NULL
  all_S2 <- NULL
  M <- 200

  with_no_grad({
    for (s in test_subjs) {
      idx <- which(stan_data$subj == s)
      N_v <- length(idx)
      x_t <- torch_tensor(X[idx, ], dtype = torch_float(), device = device)$unsqueeze(1)
      preds <- model(x_t)
      mask <- c(FALSE, rep(TRUE, length(idx) - 1))
      all_preds_switch <- c(all_preds_switch, as.numeric(preds$p$cpu())[mask])
      all_true_switch <- c(all_true_switch, Switch[idx][mask])
      all_true_rt <- c(all_true_rt, stan_data$RT[idx])

      pi_arr <- as.matrix(preds$pi[1, , ]$cpu())
      mu_arr <- as.matrix(preds$mu[1, , ]$cpu())
      sig_arr <- as.matrix(preds$sig[1, , ]$cpu())
      tau_arr <- as.matrix(preds$tau[1, , ]$cpu())

      comp1_S1 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[, 1], M))
      comp1_S2 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[, 1], M))

      mu1_M <- rep(mu_arr[, 1], M)
      sig1_M <- rep(sig_arr[, 1], M)
      tau1_M <- rep(tau_arr[, 1], M)

      mu2_M <- rep(mu_arr[, 2], M)
      sig2_M <- rep(sig_arr[, 2], M)
      tau2_M <- rep(tau_arr[, 2], M)

      samp_S1 <- ifelse(comp1_S1 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
      samp_S2 <- ifelse(comp1_S2 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)

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

  roc_obj <- roc(all_true_switch, all_preds_switch, direction = "<", quiet = TRUE)
  cv_auc[f] <- as.numeric(auc(roc_obj))
  cv_pr_auc[f] <- calculate_pr_auc(all_preds_switch, all_true_switch)
  preds_binary <- ifelse(all_preds_switch > 0.5, 1, 0)
  cm <- table(Prediction = preds_binary, Reference = all_true_switch)
  cv_acc[f] <- sum(diag(cm)) / sum(cm)
  cv_crps[f] <- mean(rowMeans(abs(all_S1 - all_true_rt)) - 0.5 * rowMeans(abs(all_S1 - all_S2)))

  cat(sprintf("fold %d results... roc-auc: %.4f, pr-auc: %.4f, crps: %.4f, acc: %.4f\n", f, cv_auc[f], cv_pr_auc[f], cv_crps[f], cv_acc[f]))
}

# save results
fit_results <- tibble(
  cv_auc = cv_auc,
  cv_pr_auc = cv_pr_auc,
  cv_crps = cv_crps,
  cv_acc = cv_acc
)
cli::cli_alert_info("saving fit results")
write_rds(fit_results, "../../results/rnn_results.rds")

cli::cli_alert_info("final LNSO cv metrics")
cli::cli_text(sprintf("mean roc-auc: %.4f (%.4f)", mean(cv_auc), sd(cv_auc)))
cli::cli_text(sprintf("mean pr-auc: %.4f (%.4f)", mean(cv_pr_auc), sd(cv_pr_auc)))
cli::cli_text(sprintf("mean crps: %.4f (%.4f)", mean(cv_crps), sd(cv_crps)))
cli::cli_text(sprintf("mean acc: %.4f (%.4f)", mean(cv_acc), sd(cv_acc)))

cli::cli_alert_info("retraining final frozen model on all 50 isolated participants")
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

    mask <- c(FALSE, rep(TRUE, length(idx) - 1))
    if (sum(mask) > 0) {
      p_t <- preds$p[, mask, , drop = FALSE]
      y_t <- y_switch[, mask, , drop = FALSE]
      n_switch <- y_t$sum()$item()
      n_stay <- y_t$numel() - n_switch
      pos_weight <- if (n_switch > 0) n_stay / n_switch else 1.0
      bce <- -(y_t * torch_log(p_t + 1e-7) * torch_tensor(pos_weight, device = device) + (1 - y_t) * torch_log(1 - p_t + 1e-7))
      loss_p <- bce$mean()
    } else {
      loss_p <- torch_tensor(0, device = device)
    }
    log_y <- torch_log(y_rt - preds$tau + 1e-6)
    log_p_k <- -log_y - 0.5 * log(2 * pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
    log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
    loss_r <- -log_mix$mean()

    total_loss <- torch_exp(-torch_clamp(model_final$log_var_policy, min = -5, max = 5)) * loss_p + model_final$log_var_policy + torch_exp(-torch_clamp(model_final$log_var_kin, min = -5, max = 5)) * loss_r + model_final$log_var_kin
    total_loss$backward()
    nn_utils_clip_grad_norm_(model_final$parameters, max_norm = 1.0)
    optimizer$step()
  }
}


cli::cli_alert_info("saving frozen final model...")
for (p in model_final$parameters) {
  p$requires_grad_(FALSE)
}
torch_save(model_final, "../../results/frozen_rnn_baseline.pt")
cli::cli_alert_success("final frozen model saved!")
