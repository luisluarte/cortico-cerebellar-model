library(cmdstanr)
library(jsonlite)
library(pROC)

cat("=== Phase 4: Baseline Models LNSO OOB Evaluation ===\n")

# Load Data
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
stan_data <- as.data.frame(stan_data)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)

mask <- stan_data$subj %in% scaffold_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)

Switch <- numeric(N_trials)
current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
  }
}
Y_switch <- Switch

# Helper Metrics
calculate_pr_auc <- function(probs, labels) {
  ord <- order(probs, decreasing = TRUE)
  probs <- probs[ord]; labels <- labels[ord]
  tp <- cumsum(labels == 1); fp <- cumsum(labels == 0)
  precision <- tp / (tp + fp); recall <- tp / sum(labels == 1)
  recall_diff <- c(recall[1], diff(recall))
  sum(recall_diff * precision, na.rm = TRUE)
}

calc_metrics <- function(probs, labels) {
  roc <- as.numeric(pROC::auc(labels, probs, quiet=TRUE))
  pr <- calculate_pr_auc(probs, labels)
  preds <- ifelse(probs > 0.5, 1, 0)
  TP <- sum(preds == 1 & labels == 1); TN <- sum(preds == 0 & labels == 0)
  FP <- sum(preds == 1 & labels == 0); FN <- sum(preds == 0 & labels == 1)
  N <- length(labels)
  num <- (TP * TN) - (FP * FN)
  den <- sqrt(as.numeric(TP+FP) * as.numeric(TP+FN) * as.numeric(TN+FP) * as.numeric(TN+FN))
  mcc <- if (den == 0) 0 else num / den
  list(roc=roc, pr=pr, mcc=mcc, tp=TP/N, tn=TN/N, fp=FP/N, fn=FN/N)
}

# 1. Deterministic WSLS (Evaluated across all 10 folds immediately)
cat("\n--- Evaluating Deterministic WSLS ---\n")
wsls_p_switch <- numeric(N_trials)
for(i in 1:N_trials) {
  if (i == 1 || stan_data$subj[i] != stan_data$subj[i-1]) {
    wsls_p_switch[i] <- 0.5
  } else {
    wsls_p_switch[i] <- ifelse(stan_data$Reward[i-1] > 0.5, 0.0, 1.0)
  }
}
folds <- split(scaffold_subjs, ceiling(seq_along(scaffold_subjs)/5))
wsls_metrics <- list(roc=c(), pr=c(), mcc=c(), tp=c(), tn=c(), fp=c(), fn=c())
for(f in 1:length(folds)) {
  test_idx <- which(stan_data$subj %in% folds[[f]])
  m <- calc_metrics(wsls_p_switch[test_idx], Y_switch[test_idx])
  for(k in names(m)) wsls_metrics[[k]] <- c(wsls_metrics[[k]], m[[k]])
}
cat(sprintf("WSLS | ROC: %.4f | PR: %.4f | MCC: %.4f | TP:%.3f TN:%.3f FP:%.3f FN:%.3f\n", 
    mean(wsls_metrics$roc), mean(wsls_metrics$pr), mean(wsls_metrics$mcc),
    mean(wsls_metrics$tp), mean(wsls_metrics$tn), mean(wsls_metrics$fp), mean(wsls_metrics$fn)))


# 2. Q-Learning with CF (Stan MCMC)
cat("\n--- Evaluating Q-Learning (Stan MCMC) ---\n")
mod <- cmdstan_model("q_learning_cf.stan")

q_metrics <- list(roc=c(), pr=c(), mcc=c(), tp=c(), tn=c(), fp=c(), fn=c())

for(f in 1:length(folds)) {
  cat(sprintf("Fold %d / 10\n", f))
  train_subjs <- setdiff(scaffold_subjs, folds[[f]])
  test_subjs <- folds[[f]]
  
  train_idx <- which(stan_data$subj %in% train_subjs)
  
  # Remap subjects 1..45
  subj_map <- as.numeric(factor(stan_data$subj[train_idx]))
  
  stan_data_train <- list(
    N = length(train_idx),
    S = length(unique(subj_map)),
    subj = subj_map,
    bd1 = stan_data$Bd1[train_idx],
    bd2 = stan_data$Bd2[train_idx],
    resp = stan_data$Resp[train_idx],
    reward = stan_data$Reward[train_idx]
  )
  
  # Fit MCMC
  fit <- mod$sample(
    data = stan_data_train,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 500,
    refresh = 0,
    show_messages = FALSE
  )
  
  # Extract population means
  draws <- fit$draws(c("mu_aw_raw", "mu_al_raw", "mu_beta_raw"), format="df")
  pop_aw <- plogis(mean(draws$mu_aw_raw))
  pop_al <- plogis(mean(draws$mu_al_raw))
  pop_beta <- exp(mean(draws$mu_beta_raw))
  
  # Test Phase
  test_idx <- which(stan_data$subj %in% test_subjs)
  test_data <- stan_data[test_idx, ]
  
  p_switch_test <- numeric(nrow(test_data))
  Q <- rep(0.5, 8)
  prev_ch <- -1
  
  for (i in 1:nrow(test_data)) {
    if (i == 1 || test_data$subj[i] != test_data$subj[i-1]) {
      Q <- rep(0.5, 8)
      prev_ch <- -1
    }
    
    b1 <- test_data$Bd1[i]
    b2 <- test_data$Bd2[i]
    
    p_b1 <- plogis(pop_beta * (Q[b1] - Q[b2]))
    p_b2 <- 1 - p_b1
    
    if (prev_ch == b1) {
      p_switch_test[i] <- p_b2
    } else if (prev_ch == b2) {
      p_switch_test[i] <- p_b1
    } else {
      p_switch_test[i] <- 1.0 # Forced switch
    }
    
    # Update Q
    c_idx <- ifelse(test_data$Resp[i] == 1, b1, b2)
    u_idx <- ifelse(test_data$Resp[i] == 1, b2, b1)
    r <- test_data$Reward[i]
    r_cf <- 1 - r
    
    if (r > 0.5) Q[c_idx] <- Q[c_idx] + pop_aw * (r - Q[c_idx])
    else Q[c_idx] <- Q[c_idx] + pop_al * (r - Q[c_idx])
    
    if (r_cf > 0.5) Q[u_idx] <- Q[u_idx] + pop_aw * (r_cf - Q[u_idx])
    else Q[u_idx] <- Q[u_idx] + pop_al * (r_cf - Q[u_idx])
    
    prev_ch <- c_idx
  }
  
  m <- calc_metrics(p_switch_test, Y_switch[test_idx])
  for(k in names(m)) q_metrics[[k]] <- c(q_metrics[[k]], m[[k]])
}

cat("\n--- Final Q-Learning Metrics (Mean) ---\n")
cat(sprintf("Q-Learn | ROC: %.4f | PR: %.4f | MCC: %.4f | TP:%.3f TN:%.3f FP:%.3f FN:%.3f\n", 
    mean(q_metrics$roc), mean(q_metrics$pr), mean(q_metrics$mcc),
    mean(q_metrics$tp), mean(q_metrics$tn), mean(q_metrics$fp), mean(q_metrics$fn)))

cat("=== Baselines Evaluation Complete ===\n")
