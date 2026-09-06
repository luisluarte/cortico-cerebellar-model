library(mco)
library(jsonlite)
library(torch)
library(pROC)

cat("Loading Scaffold Data for LNSO...\n")
stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
mask <- stan_data$subj %in% scaffold_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)

Switch <- numeric(N_trials)
Ch <- ifelse(stan_data$Resp == 1, stan_data$Bd1, stan_data$Bd2)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
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
ITI <- c(5.0, rep(0.0, N_trials - 1))
current_subj <- -1
for(i in 1:N_trials) {
    if (stan_data$subj[i] != current_subj) { ITI[i] <- 5.0; current_subj <- stan_data$subj[i] }
}
Y_switch <- Switch
Y_RT <- stan_data$RT

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

cat("Loading RNN Oracle...\n")
SpatialRNN <- nn_module("SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$pi_head <- nn_linear(hidden_dim, K)
    self$mu_head <- nn_linear(hidden_dim, K)
    self$sigma_head <- nn_linear(hidden_dim, K)
    self$tau_head <- nn_linear(hidden_dim, K)
  },
  forward = function(x, h0 = NULL) {
    h <- self$gru(x, h0)[[1]]
    list(p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)), 
         mu = self$mu_head(h), sigma = nnf_softplus(torch_clamp(self$sigma_head(h), min=-15, max=15)) + 1e-4, 
         pi = nnf_softmax(torch_clamp(self$pi_head(h), min=-15, max=15), dim=-1))
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

with_no_grad({
  preds <- model_rnn(torch_tensor(X, dtype=torch_float())$unsqueeze(1))
  p_val <- as.numeric(preds$p$squeeze())
  p_val[p_val < 1e-7] <- 1e-7; p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
  rnn_mu <- as.matrix(preds$mu$squeeze())
  rnn_sigma <- as.matrix(preds$sigma$squeeze())
  rnn_pi <- as.matrix(preds$pi$squeeze())
})

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

crps_normal <- function(y, mu, sigma) {
  z <- (y - mu) / sigma
  sigma * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi))
}

# 10-Fold CV setup
unique_subjs <- unique(stan_data$subj)
create_folds <- function(subjs, k=10) {
  set.seed(42)
  shuffled <- sample(subjs)
  splits <- split(shuffled, rep(1:k, length.out=length(shuffled)))
  lapply(splits, function(s) setdiff(subjs, s))
}
folds <- create_folds(unique_subjs, k = 10)

evaluate_checkpoint <- function(model_name, theta, gamma) {
  metrics <- list(roc=c(), pr=c(), mcc=c(), f1=c(), crps=c())
  
  if (model_name == "WSLS") {
    theta_win <- max(1e-7, min(1-1e-7, theta[1])); theta_loss <- max(1e-7, min(1-1e-7, theta[2]))
    wsls_p <- ifelse(lag_Reward > 0, theta_win, theta_loss); wsls_p[lag_Reward == 0 & lag_Ch == 0] <- 0.5 
    bio_logits <- qlogis(wsls_p)
    State <- matrix(lag_Reward, ncol=1)
  } else if (model_name == "QLearn") {
    alpha_win <- theta[1]; alpha_loss <- theta[2]; beta <- theta[3]
    Q <- rep(0.5, 8); bio_logits <- numeric(N_trials); State <- matrix(0, nrow=N_trials, ncol=8); current_s <- -1
    for (t in 1:N_trials) {
      if (stan_data$subj[t] != current_s) { Q <- rep(0.5, 8); current_s <- stan_data$subj[t] }
      State[t, ] <- Q
      b1 <- stan_data$Bd1[t]; b2 <- stan_data$Bd2[t]
      p_bd1 <- plogis(beta * (Q[b1] - Q[b2]))
      if (lag_Resp[t] == 1) { p_switch <- ifelse(lag_Ch[t] == b1, 1-p_bd1, p_bd1) } else { p_switch <- ifelse(lag_Ch[t] == b1, 1-p_bd1, p_bd1) }
      p_switch <- max(1e-7, min(1-1e-7, p_switch))
      bio_logits[t] <- qlogis(p_switch)
      chosen <- ifelse(stan_data$Resp[t] == 1, b1, b2); unchosen <- ifelse(stan_data$Resp[t] == 1, b2, b1)
      if (stan_data$Reward[t] > 0) { Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen]); Q[unchosen] <- Q[unchosen] + alpha_win * (0 - Q[unchosen])
      } else { Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen]); Q[unchosen] <- Q[unchosen] + alpha_loss * (1 - Q[unchosen]) }
    }
  } else if (model_name == "Bio") {
    p_a <- theta[1]; p_l <- theta[2]; p_bt <- theta[3]; p_k <- theta[4]; p_ag <- theta[5]; p_bg <- theta[6]; p_sd <- theta[7]
    mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
    State <- matrix(0, nrow=N_trials, ncol=32); current_s <- -1
    for (t in 1:N_trials) {
      if (ITI[t] > 0) { Z <- rep(0, 896); W_purk <- W_purk + rnorm(896, 0, sqrt(p_sd * ITI[t])) }
      I_t <- X[t, ]; I_hat <- as.numeric(W_gen %*% mu); eps <- Pi_vec * (I_t - I_hat)
      mu <- mu + p_a * (as.numeric(t(W_gen) %*% eps) - (p_l * ITI[t]) * mu + p_bt * as.numeric(W_thal %*% D))
      G <- as.numeric(W_ach1 %*% mu); Z <- (1.0 - p_bg) * Z + p_ag * G; eps_mag <- mean(abs(eps))
      W_purk <- W_purk - p_k * (eps_mag * Z); D <- as.numeric(W_ach2 %*% (G * W_purk))
      State[t, ] <- mu
    }
  }
  
  pr_auc <- function(probs, labels) {
  ord <- order(probs, decreasing=TRUE)
  p <- probs[ord]; l <- labels[ord]
  tp <- cumsum(l == 1); fp <- cumsum(l == 0)
  precision <- tp / (tp + fp); recall <- tp / sum(l == 1)
  precision[is.na(precision)] <- 1
  auc <- sum(diff(recall) * (precision[-1] + precision[-length(precision)]) / 2)
  return(auc)
}

# Cross-validation
  for (fold in 1:10) {
    train_subjs <- unique_subjs[folds[[fold]]]
    train_idx <- stan_data$subj %in% train_subjs
    test_idx <- !train_idx
    
    State_train <- State[train_idx, , drop=FALSE]
    State_test <- State[test_idx, , drop=FALSE]
    
    XX_inv <- tryCatch(solve(crossprod(State_train) + diag(0.01, ncol(State_train))), error=function(e) NULL)
    if (is.null(XX_inv)) return(NULL)
    
    W_mu <- XX_inv %*% crossprod(State_train, rnn_mu[train_idx, , drop=FALSE])
    
    if (model_name == "Bio") {
      W_policy <- XX_inv %*% crossprod(State_train, rnn_p_logits[train_idx])
      test_bio_logits <- as.numeric(State_test %*% W_policy)
    } else {
      test_bio_logits <- bio_logits[test_idx]
    }
    
    blend_logits <- (1 - gamma) * rnn_p_logits[test_idx] + gamma * test_bio_logits
    test_probs <- plogis(blend_logits)
    
    test_bio_mu <- State_test %*% W_mu
    blend_mu <- (1 - gamma) * rnn_mu[test_idx, , drop=FALSE] + gamma * test_bio_mu
    
    roc <- roc(Y_switch[test_idx], test_probs, quiet=TRUE)$auc
    pr <- pr_auc(test_probs, Y_switch[test_idx])
    pred_class <- ifelse(test_probs > 0.5, 1, 0)
    tp <- sum(pred_class == 1 & Y_switch[test_idx] == 1); tn <- sum(pred_class == 0 & Y_switch[test_idx] == 0)
    fp <- sum(pred_class == 1 & Y_switch[test_idx] == 0); fn <- sum(pred_class == 0 & Y_switch[test_idx] == 1)
    mcc <- (tp*tn - fp*fn) / sqrt(pmax((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1e-6))
    f1 <- 2*tp / (2*tp + fp + fn)
    
    pi_t <- rnn_pi[test_idx, , drop=FALSE]; sigma_t <- rnn_sigma[test_idx, , drop=FALSE]
    mu_m <- pi_t[,1]*blend_mu[,1] + pi_t[,2]*blend_mu[,2]
    var_m <- pi_t[,1]*(sigma_t[,1]^2 + blend_mu[,1]^2) + pi_t[,2]*(sigma_t[,2]^2 + blend_mu[,2]^2) - mu_m^2
    sigma_m <- sqrt(pmax(var_m, 1e-6))
    crps <- mean(crps_normal(Y_RT[test_idx], mu_m, sigma_m))
    
    metrics$roc <- c(metrics$roc, roc); metrics$pr <- c(metrics$pr, pr); metrics$mcc <- c(metrics$mcc, mcc)
    metrics$f1 <- c(metrics$f1, f1); metrics$crps <- c(metrics$crps, crps)
  }
  
  return(data.frame(Model=model_name, Gamma=gamma, ROC=mean(metrics$roc), PR=mean(metrics$pr), 
                    MCC=mean(metrics$mcc), F1=mean(metrics$f1), CRPS=mean(metrics$crps)))
}

cat("Loading Master Pareto Fronts...\n")
res <- readRDS("master_pareto_fronts.rds")
out_results <- data.frame()

for (m in c("WSLS", "QLearn", "Bio")) {
  df <- res[res$Model == m, ]
  df <- df[order(df$Gamma), ]
  
  idx_min <- 1
  idx_max <- nrow(df)
  idx_med <- round(nrow(df) / 2)
  
  for (idx in c(idx_min, idx_med, idx_max)) {
    cat(sprintf("Evaluating %s at Gamma = %.3f...\n", m, df$Gamma[idx]))
    metrics <- evaluate_checkpoint(m, df$Params[[idx]], df$Gamma[idx])
    if (!is.null(metrics)) out_results <- rbind(out_results, metrics)
  }
}

print(out_results)
write.csv(out_results, "master_lnso_results.csv", row.names=FALSE)
cat("LNSO Benchmark Complete.\n")
