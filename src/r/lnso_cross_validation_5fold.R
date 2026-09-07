local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
  library(mco)
  library(doParallel)
  library(foreach)
  library(tibble)
  library(dplyr)
  library(Rcpp)
  library(RcppArmadillo)
})

cat("Compiling C++ Bio Simulator on Main Process...\n")
sourceCpp("sim_bio.cpp")

cat("Loading Distillation Targets 2.0...\n")
targets <- readRDS("distillation_targets_v2.rds")

# Global Platt Scaling Calibration
fit <- glm(targets$labels ~ targets$rnn_logits, family = binomial)
calibrated_rnn_logits <- predict(fit, type = "link")
teacher_soft_targets <- plogis(calibrated_rnn_logits)

# Bio Initialization
set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

POPSIZE <- 128
GENERATIONS <- 40

cat("Starting Parallel Cluster on 32 Cores...\n")
cl <- makeCluster(32)
registerDoParallel(cl)

cat("Compiling C++ Bio Simulator on all 32 Cores...\n")
clusterEvalQ(cl, {
  library(Rcpp)
  library(RcppArmadillo)
  sourceCpp("sim_bio.cpp")
})

# Cross Validation Setup
unique_subjs <- unique(targets$subjs)
set.seed(123)
shuffled_subjs <- sample(unique_subjs)
folds <- split(shuffled_subjs, ceiling(seq_along(shuffled_subjs) / 10)) # 5 folds of 10 subjects

all_cv_results <- list()

for (f in 1:5) {
  cat(sprintf("\n========================================\n"))
  cat(sprintf("STARTING FOLD %d / 5\n", f))
  cat(sprintf("========================================\n"))
  
  test_subjs <- folds[[f]]
  train_subjs <- setdiff(unique_subjs, test_subjs)
  
  train_idx <- which(targets$subjs %in% train_subjs)
  test_idx <- which(targets$subjs %in% test_subjs)
  
  # ================= TRAIN PHASE =================
  cat("Extracting Train Data...\n")
  t_N <- length(train_idx)
  t_X <- targets$X[train_idx, ]
  t_ITI <- targets$ITI[train_idx]
  t_lag_R <- targets$lag_reward[train_idx]
  t_lag_C <- targets$lag_ch[train_idx]
  t_lag_Resp <- targets$lag_resp[train_idx]
  t_subjs <- targets$subjs[train_idx]
  
  t_rnn_logits <- calibrated_rnn_logits[train_idx]
  t_p_val <- teacher_soft_targets[train_idx]
  t_rnn_mu <- targets$rnn_mu[train_idx, ]
  t_rnn_sigma <- targets$rnn_sigma[train_idx, ]
  
  cat("Training WSLS (NSGA-II)...\n")
  wsls_obj <- function(params_matrix) {
    if(nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
    foreach(ind = 1:ncol(params_matrix), .combine = cbind, .export=c("t_lag_R", "t_lag_C", "t_rnn_logits", "t_p_val", "t_rnn_mu", "t_rnn_sigma")) %dopar% {
      theta_win <- max(1e-7, min(1-1e-7, params_matrix[1, ind]))
      theta_loss <- max(1e-7, min(1-1e-7, params_matrix[2, ind]))
      gamma <- params_matrix[3, ind]
      wsls_p <- ifelse(t_lag_R > 0, theta_win, theta_loss)
      wsls_p[t_lag_R == 0 & t_lag_C == 0] <- 0.5 
      blend_logits <- (1 - gamma) * t_rnn_logits + gamma * qlogis(wsls_p)
      blend_probs <- pmax(pmin(plogis(blend_logits), 1 - 1e-7), 1e-7)
      bce <- -mean(t_p_val * log(blend_probs) + (1 - t_p_val) * log(1 - blend_probs))
      State <- matrix(t_lag_R, ncol=1)
      XX_inv <- solve(crossprod(State) + diag(0.01, 1))
      W_mu <- XX_inv %*% crossprod(State, t_rnn_mu)
      blend_mu <- (1 - gamma) * t_rnn_mu + gamma * (State %*% W_mu)
      kinematic_loss <- mean((t_rnn_mu - blend_mu)^2 / (2 * t_rnn_sigma^2))
      c(bce + kinematic_loss, 1.0 - gamma)
    }
  }
  res_wsls <- nsga2(wsls_obj, idim = 3, odim = 2, lower.bounds = c(0.001, 0.001, 0.0), upper.bounds = c(0.999, 0.999, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
  
  cat("Training Q-Learning (NSGA-II)...\n")
  q_learning_obj <- function(params_matrix) {
    if(nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
    foreach(ind = 1:ncol(params_matrix), .combine = cbind, .export=c("t_N", "t_lag_Resp", "t_lag_C", "t_rnn_logits", "t_p_val", "t_rnn_mu", "t_rnn_sigma", "t_X", "t_subjs", "t_lag_R")) %dopar% {
      alpha_win <- params_matrix[1, ind]; alpha_loss <- params_matrix[2, ind]; beta <- params_matrix[3, ind]; gamma <- params_matrix[4, ind]
      Q <- rep(0.5, 8); q_logits <- numeric(t_N); Q_history <- matrix(0, nrow=t_N, ncol=8)
      bd1 <- max.col(t_X[, 1:8]); bd2 <- max.col(t_X[, 9:16])
      current_subj <- -1
      for (t in 1:t_N) {
        if (t_subjs[t] != current_subj) { Q <- rep(0.5, 8); current_subj <- t_subjs[t] }
        Q_history[t, ] <- Q
        b1 <- bd1[t]; b2 <- bd2[t]
        p_bd1 <- plogis(beta * (Q[b1] - Q[b2]))
        p_switch <- ifelse(t_lag_Resp[t] == 1, ifelse(t_lag_C[t] == b1, 1 - p_bd1, p_bd1), ifelse(t_lag_C[t] == b1, 1 - p_bd1, p_bd1))
        q_logits[t] <- qlogis(max(1e-7, min(1-1e-7, p_switch)))
        chosen <- ifelse(t_lag_Resp[t] == 1, b1, b2); unchosen <- ifelse(t_lag_Resp[t] == 1, b2, b1)
        if (t_lag_R[t] > 0) { Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen]); Q[unchosen] <- Q[unchosen] + alpha_win * (0 - Q[unchosen]) }
        else { Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen]); Q[unchosen] <- Q[unchosen] + alpha_loss * (1 - Q[unchosen]) }
      }
      blend_logits <- (1 - gamma) * t_rnn_logits + gamma * q_logits
      blend_probs <- pmax(pmin(plogis(blend_logits), 1 - 1e-7), 1e-7)
      bce <- -mean(t_p_val * log(blend_probs) + (1 - t_p_val) * log(1 - blend_probs))
      XX_inv <- tryCatch(solve(crossprod(Q_history) + diag(0.01, 8)), error = function(e) NULL)
      if(is.null(XX_inv)) { kinematic_loss <- 10.0 } else {
        blend_mu <- (1 - gamma) * t_rnn_mu + gamma * (Q_history %*% (XX_inv %*% crossprod(Q_history, t_rnn_mu)))
        kinematic_loss <- mean((t_rnn_mu - blend_mu)^2 / (2 * t_rnn_sigma^2))
      }
      c(bce + kinematic_loss, 1.0 - gamma)
    }
  }
  res_qlearn <- nsga2(q_learning_obj, idim = 4, odim = 2, lower.bounds = c(0.001, 0.001, 0.1, 0.0), upper.bounds = c(0.999, 0.999, 10.0, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)

  cat("Training Cortico-Cerebellar (NSGA-II via Rcpp)...\n")
  cortico_obj <- function(params_matrix) {
    if(nrow(params_matrix) > ncol(params_matrix)) params_matrix <- t(params_matrix)
    foreach(ind = 1:ncol(params_matrix), .combine = cbind, .export=c("t_N", "t_X", "W_gen", "W_ach1", "W_ach2", "W_thal", "Pi_vec", "t_ITI", "t_rnn_logits", "t_p_val", "t_rnn_mu", "t_rnn_sigma")) %dopar% {
      p_alpha_pc <- params_matrix[1, ind]; p_lambda_pc <- params_matrix[2, ind]; p_beta_thal <- params_matrix[3, ind]; p_kappa_cf <- params_matrix[4, ind]
      p_alpha_gran <- params_matrix[5, ind]; p_beta_gran <- params_matrix[6, ind]; p_sigma2_diff <- params_matrix[7, ind]; gamma <- params_matrix[8, ind]
      mu_history <- simulate_bio(t_N, t_X, t_ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_vec, p_alpha_pc, p_lambda_pc, p_beta_thal, p_kappa_cf, p_alpha_gran, p_beta_gran, p_sigma2_diff)
      if (nrow(mu_history) == 1 && is.na(mu_history[1,1])) { c(100.0, 1.0 - gamma) } else {
        XX_inv <- tryCatch(solve(crossprod(mu_history) + diag(0.01, 32)), error = function(e) NULL)
        if(is.null(XX_inv)) { c(100.0, 1.0 - gamma) } else {
          bio_logits <- as.numeric(mu_history %*% (XX_inv %*% crossprod(mu_history, t_rnn_logits)))
          bio_mu <- mu_history %*% (XX_inv %*% crossprod(mu_history, t_rnn_mu))
          blend_logits <- (1 - gamma) * t_rnn_logits + gamma * bio_logits
          blend_mu <- (1 - gamma) * t_rnn_mu + gamma * bio_mu
          kinematic_loss <- mean( (t_rnn_mu - blend_mu)^2 / (2 * t_rnn_sigma^2) )
          blend_probs <- pmax(pmin(plogis(blend_logits), 1 - 1e-7), 1e-7)
          bce <- -mean(t_p_val * log(blend_probs) + (1 - t_p_val) * log(1 - blend_probs))
          c(bce + kinematic_loss, 1.0 - gamma)
        }
      }
    }
  }
  res_bio <- nsga2(cortico_obj, idim = 8, odim = 2, lower.bounds = c(0.001, 0.001, 0.001, 0.0001, 0.001, 0.001, 0.001, 0.0), upper.bounds = c(1.500, 1.500, 0.500, 0.1000, 1.000, 1.000, 0.500, 1.0), popsize = POPSIZE, generations = GENERATIONS, cprob = 0.9, mprob = 0.2, vectorized = TRUE)
  
  # ================= EXTRACT QUARTILES =================
  extract_quartiles <- function(res, idim) {
    df <- as.data.frame(res$par[res$pareto.optimal, , drop=FALSE])
    colnames(df) <- paste0("p", 1:idim)
    df$gamma <- df[[paste0("p", idim)]] # Gamma is always the last parameter
    df <- df[order(df$gamma), ]
    n <- nrow(df)
    q1 <- df[max(1, floor(n * 0.25)), ]
    q2 <- df[max(1, floor(n * 0.50)), ]
    q3 <- df[max(1, floor(n * 0.75)), ]
    list(Q1 = q1, Q2 = q2, Q3 = q3)
  }
  wsls_q <- extract_quartiles(res_wsls, 3)
  qlearn_q <- extract_quartiles(res_qlearn, 4)
  bio_q <- extract_quartiles(res_bio, 8)
  
  # ================= TEST PHASE =================
  cat("Extracting Test Data...\n")
  ts_N <- length(test_idx); ts_X <- targets$X[test_idx, ]; ts_ITI <- targets$ITI[test_idx]; ts_lag_R <- targets$lag_reward[test_idx]
  ts_lag_C <- targets$lag_ch[test_idx]; ts_lag_Resp <- targets$lag_resp[test_idx]; ts_subjs <- targets$subjs[test_idx]
  ts_labels <- targets$labels[test_idx]
  ts_rnn_logits <- calibrated_rnn_logits[test_idx]; ts_p_val <- teacher_soft_targets[test_idx]
  ts_rnn_mu <- targets$rnn_mu[test_idx, ]; ts_rnn_sigma <- targets$rnn_sigma[test_idx, ]
  
  run_test <- function(model_name, quartile_name, params, eval_func) {
    gamma_train <- params$gamma
    res <- eval_func(as.numeric(params[1:(length(params)-1)]))
    
    # Pure student evaluation
    bce_emp <- -mean(ts_labels * log(res$probs) + (1 - ts_labels) * log(1 - res$probs))
    bce_teach <- -mean(ts_p_val * log(res$probs) + (1 - ts_p_val) * log(1 - res$probs))
    kinematic <- mean((ts_rnn_mu - res$mu)^2 / (2 * ts_rnn_sigma^2))
    
    tibble(
      Fold = f, Model = model_name, Quartile = quartile_name, Gamma_Train = gamma_train,
      Test_BCE_Emp = bce_emp, Test_BCE_Teacher = bce_teach, Test_Kinematic = kinematic,
      Composite_Emp = bce_emp + kinematic, Composite_Teacher = bce_teach + kinematic
    )
  }
  
  # Eval Functions (PURE STUDENT => gamma = 1.0 logic embedded)
  eval_wsls <- function(p) {
    wsls_p <- ifelse(ts_lag_R > 0, p[1], p[2])
    wsls_p[ts_lag_R == 0 & ts_lag_C == 0] <- 0.5 
    logits <- qlogis(wsls_p)
    probs <- pmax(pmin(plogis(logits), 1-1e-7), 1e-7)
    State <- matrix(ts_lag_R, ncol=1)
    # Use Train mapping for mu
    t_State <- matrix(t_lag_R, ncol=1)
    XX_inv <- solve(crossprod(t_State) + diag(0.01, 1))
    W_mu <- XX_inv %*% crossprod(t_State, t_rnn_mu)
    mu <- State %*% W_mu
    list(probs = probs, mu = mu)
  }
  
  eval_qlearn <- function(p) {
    Q <- rep(0.5, 8); logits <- numeric(ts_N); Q_history <- matrix(0, nrow=ts_N, ncol=8)
    bd1 <- max.col(ts_X[, 1:8]); bd2 <- max.col(ts_X[, 9:16])
    current_subj <- -1
    for (t in 1:ts_N) {
      if (ts_subjs[t] != current_subj) { Q <- rep(0.5, 8); current_subj <- ts_subjs[t] }
      Q_history[t, ] <- Q
      b1 <- bd1[t]; b2 <- bd2[t]
      p_bd1 <- plogis(p[3] * (Q[b1] - Q[b2]))
      p_switch <- ifelse(ts_lag_Resp[t] == 1, ifelse(ts_lag_C[t] == b1, 1 - p_bd1, p_bd1), ifelse(ts_lag_C[t] == b1, 1 - p_bd1, p_bd1))
      logits[t] <- qlogis(max(1e-7, min(1-1e-7, p_switch)))
      chosen <- ifelse(ts_lag_Resp[t] == 1, b1, b2); unchosen <- ifelse(ts_lag_Resp[t] == 1, b2, b1)
      if (ts_lag_R[t] > 0) { Q[chosen] <- Q[chosen] + p[1] * (1 - Q[chosen]); Q[unchosen] <- Q[unchosen] + p[1] * (0 - Q[unchosen]) }
      else { Q[chosen] <- Q[chosen] + p[2] * (0 - Q[chosen]); Q[unchosen] <- Q[unchosen] + p[2] * (1 - Q[unchosen]) }
    }
    probs <- pmax(pmin(plogis(logits), 1-1e-7), 1e-7)
    
    # Use Train mapping for mu (We must re-simulate Train to get Q_history_train)
    t_Q <- rep(0.5, 8); t_Q_history <- matrix(0, nrow=t_N, ncol=8)
    t_bd1 <- max.col(t_X[, 1:8]); t_bd2 <- max.col(t_X[, 9:16])
    current_subj <- -1
    for (t in 1:t_N) {
      if (t_subjs[t] != current_subj) { t_Q <- rep(0.5, 8); current_subj <- t_subjs[t] }
      t_Q_history[t, ] <- t_Q
      b1 <- t_bd1[t]; b2 <- t_bd2[t]
      chosen <- ifelse(t_lag_Resp[t] == 1, b1, b2); unchosen <- ifelse(t_lag_Resp[t] == 1, b2, b1)
      if (t_lag_R[t] > 0) { t_Q[chosen] <- t_Q[chosen] + p[1] * (1 - t_Q[chosen]); t_Q[unchosen] <- t_Q[unchosen] + p[1] * (0 - t_Q[unchosen]) }
      else { t_Q[chosen] <- t_Q[chosen] + p[2] * (0 - t_Q[chosen]); t_Q[unchosen] <- t_Q[unchosen] + p[2] * (1 - t_Q[unchosen]) }
    }
    XX_inv <- solve(crossprod(t_Q_history) + diag(0.01, 8))
    W_mu <- XX_inv %*% crossprod(t_Q_history, t_rnn_mu)
    mu <- Q_history %*% W_mu
    
    list(probs = probs, mu = mu)
  }
  
  eval_bio <- function(p) {
    mu_hist_test <- simulate_bio(ts_N, ts_X, ts_ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_vec, p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    # Must re-simulate Train to get mapping
    mu_hist_train <- simulate_bio(t_N, t_X, t_ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_vec, p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    
    XX_inv <- solve(crossprod(mu_hist_train) + diag(0.01, 32))
    W_policy <- XX_inv %*% crossprod(mu_hist_train, t_rnn_logits)
    W_mu <- XX_inv %*% crossprod(mu_hist_train, t_rnn_mu)
    
    logits <- as.numeric(mu_hist_test %*% W_policy)
    probs <- pmax(pmin(plogis(logits), 1-1e-7), 1e-7)
    mu <- mu_hist_test %*% W_mu
    
    list(probs = probs, mu = mu)
  }
  
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("WSLS", "Q1", wsls_q$Q1, eval_wsls)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("WSLS", "Q2", wsls_q$Q2, eval_wsls)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("WSLS", "Q3", wsls_q$Q3, eval_wsls)
  
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("QLearn", "Q1", qlearn_q$Q1, eval_qlearn)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("QLearn", "Q2", qlearn_q$Q2, eval_qlearn)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("QLearn", "Q3", qlearn_q$Q3, eval_qlearn)
  
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("Bio", "Q1", bio_q$Q1, eval_bio)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("Bio", "Q2", bio_q$Q2, eval_bio)
  all_cv_results[[length(all_cv_results) + 1]] <- run_test("Bio", "Q3", bio_q$Q3, eval_bio)
}

stopCluster(cl)

final_results <- bind_rows(all_cv_results)
saveRDS(final_results, "lnso_5fold_results.rds")
cat("SUCCESS: Finished writing LNSO results to lnso_5fold_results.rds\n")
