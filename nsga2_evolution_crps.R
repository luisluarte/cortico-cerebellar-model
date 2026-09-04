library(mco)
library(doParallel)
library(torch)
library(jsonlite)
library(dplyr)

# 1. Load and prep data globally
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

# Base train/test split (Fold 1)
K <- 5
set.seed(42)
folds <- sample(rep(1:K, length.out = length(subjs_50)))
test_subjs <- subjs_50[folds == 1]
train_subjs <- subjs_50[folds != 1]

# 2. Setup PSOCK Cluster for Safe Torch Multiprocessing
cl <- makeCluster(20)
registerDoParallel(cl)
clusterExport(cl, c("stan_data_50", "Switch", "X", "train_subjs", "test_subjs", "min_RT"))
clusterEvalQ(cl, {
  library(torch)
  torch_set_num_threads(1)
  
  HybridUniversalRNN <- nn_module(
    "HybridUniversalRNN",
    initialize = function(input_dim, hidden_dim, E_ratio, p) {
      self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 2, batch_first = TRUE)
      
      self$policy_head <- nn_linear(hidden_dim, 1)
      self$mu_head <- nn_linear(hidden_dim, 2)
      self$sigma_head <- nn_linear(hidden_dim, 2)
      self$tau_head <- nn_linear(hidden_dim, 2)
      self$pi_head <- nn_linear(hidden_dim, 2)
      
      N_granule <- as.integer(round(hidden_dim * E_ratio))
      
      elements <- sample(c(-sqrt(3)/sqrt(hidden_dim), 0, sqrt(3)/sqrt(hidden_dim)), 
                         N_granule * hidden_dim, replace=TRUE, prob=c(p/2, 1-p, p/2))
      W_mat <- matrix(elements, nrow=hidden_dim, ncol=N_granule)
      self$W_res <- torch_tensor(W_mat, dtype=torch_float(), requires_grad=FALSE)
      
      self$pc_policy_head <- nn_linear(N_granule, 1)
      self$pc_mu_head <- nn_linear(N_granule, 2)
      self$pc_sigma_head <- nn_linear(N_granule, 2)
      self$pc_tau_head <- nn_linear(N_granule, 2)
      self$pc_pi_head <- nn_linear(N_granule, 2)
      
      self$log_var_policy <- nn_parameter(torch_zeros(1))
      self$log_var_kin <- nn_parameter(torch_zeros(1))
    },
    forward = function(x, h0 = NULL) {
      out <- self$gru(x, h0)
      h <- out[[1]]
      
      z <- torch_matmul(h$detach(), self$W_res)
      
      logits_p <- torch_clamp(self$policy_head(h) + self$pc_policy_head(z), min = -15, max = 15)
      p_switch <- nnf_sigmoid(logits_p)
      
      mu_rt <- self$mu_head(h) + self$pc_mu_head(z)
      
      logits_sigma <- torch_clamp(self$sigma_head(h) + self$pc_sigma_head(z), min = -15, max = 15)
      sigma_rt <- nnf_softplus(logits_sigma) + 1e-4
      
      logits_tau <- torch_clamp(self$tau_head(h) + self$pc_tau_head(z), min = -15, max = 15)
      tau_rt <- nnf_sigmoid(logits_tau) * (0.99 * min_RT)
      
      logits_pi <- torch_clamp(self$pi_head(h) + self$pc_pi_head(z), min = -15, max = 15)
      pi_mix <- nnf_softmax(logits_pi, dim = -1)
      
      list(p = p_switch, mu = mu_rt, sigma = sigma_rt, tau = tau_rt, pi = pi_mix, h = h)
    }
  )
})

# 3. Vectorized Fitness Function
evaluate_population <- function(params_matrix) {
  # Save checkpoint of current genotype matrix
  write.table(params_matrix, "nsga2_checkpoint_pop.csv", append = TRUE, col.names = FALSE, row.names = FALSE, sep=",")
  
  results <- foreach(i = 1:nrow(params_matrix), .combine = rbind, .packages = c("torch")) %dopar% {
    E_ratio <- round(params_matrix[i, 1])
    p <- params_matrix[i, 2]
    
    input_dim <- ncol(X)
    hidden_dim <- 32
    
    model <- HybridUniversalRNN(input_dim, hidden_dim, E_ratio, p)
    optimizer <- optim_adam(model$parameters, lr = 0.01)
    
    # Train 15 epochs
    for (ep in 1:15) {
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
    
    # Eval (CRPS via Energy Score Approximation)
    model$eval()
    M <- 1000
    crps_all <- numeric()
    
    with_no_grad({
      for (s in test_subjs) {
        idx <- which(stan_data_50$subj == s)
        N_v <- length(idx)
        x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
        preds <- model(x_t)
        
        pi_arr <- as.matrix(preds$pi[1, , ])
        mu_arr <- as.matrix(preds$mu[1, , ])
        sig_arr <- as.matrix(preds$sigma[1, , ])
        tau_arr <- as.matrix(preds$tau[1, , ])
        
        y_v <- stan_data_50$RT[idx]
        
        comp1_S1 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
        comp1_S2 <- rbinom(N_v * M, size = 1, prob = rep(pi_arr[,1], M))
        
        mu1_M <- rep(mu_arr[,1], M); sig1_M <- rep(sig_arr[,1], M); tau1_M <- rep(tau_arr[,1], M)
        mu2_M <- rep(mu_arr[,2], M); sig2_M <- rep(sig_arr[,2], M); tau2_M <- rep(tau_arr[,2], M)
        
        samp_S1 <- ifelse(comp1_S1 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
        samp_S2 <- ifelse(comp1_S2 == 1, rlnorm(N_v * M, mu1_M, sig1_M) + tau1_M, rlnorm(N_v * M, mu2_M, sig2_M) + tau2_M)
        
        mat_S1 <- matrix(samp_S1, nrow = N_v, ncol = M, byrow = FALSE)
        mat_S2 <- matrix(samp_S2, nrow = N_v, ncol = M, byrow = FALSE)
        
        E_Xy <- rowMeans(abs(mat_S1 - y_v))
        E_XX <- rowMeans(abs(mat_S1 - mat_S2))
        subj_crps <- E_Xy - 0.5 * E_XX
        crps_all <- c(crps_all, subj_crps)
      }
    })
    
    mean_crps <- mean(crps_all)
    if (is.nan(mean_crps) || is.na(mean_crps) || mean_crps < 0 || mean_crps > 10) mean_crps <- 10
    
    # Pathway Reliance L2 Ratio
    w_rnn <- c(
      as.numeric(model$policy_head$weight), as.numeric(model$mu_head$weight),
      as.numeric(model$sigma_head$weight), as.numeric(model$tau_head$weight), as.numeric(model$pi_head$weight)
    )
    w_pc <- c(
      as.numeric(model$pc_policy_head$weight), as.numeric(model$pc_mu_head$weight),
      as.numeric(model$pc_sigma_head$weight), as.numeric(model$pc_tau_head$weight), as.numeric(model$pc_pi_head$weight)
    )
    norm_rnn <- sum(w_rnn^2)
    norm_pc <- sum(w_pc^2)
    reliance <- norm_pc / (norm_pc + norm_rnn + 1e-8)
    
    # Save exact individual result to checkpoint
    res_line <- data.frame(E = E_ratio, p = p, CRPS = mean_crps, Reliance = reliance)
    write.table(res_line, "nsga2_checkpoint_results.csv", append = TRUE, col.names = FALSE, row.names = FALSE, sep=",")
    
    # NSGA-II minimizes all. Maximize reliance = minimize -reliance
    c(mean_crps, -reliance)
  }
  
  cat("Generation evaluated.\\n")
  return(t(results))
}

# 4. Execute NSGA-II
cat("Launching 20-core parallel NSGA-II...\\n")
# clean old checkpoints
if (file.exists("nsga2_checkpoint_pop.csv")) file.remove("nsga2_checkpoint_pop.csv")
if (file.exists("nsga2_checkpoint_results.csv")) file.remove("nsga2_checkpoint_results.csv")

res <- nsga2(evaluate_population, idim = 2, odim = 2, 
             lower.bounds = c(2, 0.01), upper.bounds = c(30, 0.33), 
             popsize = 20, generations = 15, vectorized = TRUE)

stopCluster(cl)
saveRDS(res, 'nsga2_crps_final.rds')
cat("EVOLUTION COMPLETE.\\n")
