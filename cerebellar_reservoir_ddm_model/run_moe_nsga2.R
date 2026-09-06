library(mco)
library(doParallel)
library(torch)
library(jsonlite)
library(dplyr)

cores <- 8
torch_set_num_threads(cores)

# 1. Load Data
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
min_RT <- min(stan_data$RT)
train_subjs <- readRDS('excluded_subjects.rds')
all_subjs <- unique(stan_data$subj)
oob_subjs <- setdiff(all_subjs, train_subjs)

N_trials <- stan_data$N
Ch <- numeric(N_trials)
for(i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials); lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
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

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# ==========================================
# PHASE 1: Train Frozen Baseline Spatial RNN
# ==========================================
SpatialRNN <- nn_module(
  "SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$pi_head <- nn_linear(hidden_dim, K)
    self$mu_head <- nn_linear(hidden_dim, K)
    self$sigma_head <- nn_linear(hidden_dim, K)
    self$tau_head <- nn_linear(hidden_dim, K)
    
    self$log_var_policy <- nn_parameter(torch_zeros(1))
    self$log_var_kin <- nn_parameter(torch_zeros(1))
    self$K <- K
  },
  forward = function(x, h0 = NULL) {
    out <- self$gru(x, h0)
    h <- out[[1]]
    
    # Pre-activation logits
    logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
    logits_pi <- torch_clamp(self$pi_head(h), min=-15, max=15)
    mu_rt <- self$mu_head(h)
    logits_sigma <- torch_clamp(self$sigma_head(h), min=-15, max=15)
    logits_tau <- torch_clamp(self$tau_head(h), min=-15, max=15)
    
    list(h = h, logits_p = logits_p, logits_pi = logits_pi, mu_rt = mu_rt, logits_sigma = logits_sigma, logits_tau = logits_tau)
  }
)

torch_manual_seed(42)
rnn_model <- SpatialRNN(ncol(X), 4, 2)
optimizer <- optim_adam(rnn_model$parameters, lr = 0.002, weight_decay = 0.0)

cat("Phase 1: Training the Pure Spatial RNN Baseline...\n")
for (ep in 1:15) {
  rnn_model$train()
  for (s in train_subjs) {
    idx <- which(stan_data$subj == s)
    x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
    y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
    y_rt <- torch_tensor(stan_data$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
    
    optimizer$zero_grad()
    preds <- rnn_model(x_t)
    
    p_switch <- nnf_sigmoid(preds$logits_p)
    pi_mix <- nnf_softmax(preds$logits_pi, dim=-1)
    sigma_rt <- nnf_softplus(preds$logits_sigma) + 1e-4
    tau_rt <- nnf_sigmoid(preds$logits_tau) * (0.99 * min_RT)
    
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    if (sum(mask) > 0) {
      p_t <- p_switch[, mask, , drop=FALSE]
      y_t <- y_switch[, mask, , drop=FALSE]
      n_switch <- y_t$sum()$item()
      n_stay <- y_t$numel() - n_switch
      pos_weight <- if (n_switch > 0) n_stay / n_switch else 1.0
      bce <- -(y_t * torch_log(p_t + 1e-7) * torch_tensor(pos_weight, device = p_t$device) + (1 - y_t) * torch_log(1 - p_t + 1e-7))
      loss_p <- bce$mean()
    } else {
      loss_p <- torch_tensor(0)
    }
    
    log_y <- torch_log(y_rt - tau_rt + 1e-6)
    log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(sigma_rt) - 0.5 * torch_square((log_y - preds$mu_rt) / sigma_rt)
    log_mix <- torch_logsumexp(torch_log(pi_mix) + log_p_k, dim = -1)
    loss_r <- -log_mix$mean()
    
    total_loss <- torch_exp(-torch_clamp(rnn_model$log_var_policy, min=-5, max=5)) * loss_p + rnn_model$log_var_policy + torch_exp(-torch_clamp(rnn_model$log_var_kin, min=-5, max=5)) * loss_r + rnn_model$log_var_kin
    
    total_loss$backward()
    nn_utils_clip_grad_norm_(rnn_model$parameters, max_norm = 1.0)
    optimizer$step()
  }
}
cat("Phase 1: Freezing RNN...\n")
rnn_model$requires_grad_(FALSE)
rnn_model$eval()
torch_save(rnn_model, "frozen_rnn.pt")

# ==========================================
# PHASE 2: MoE Cortico-Cerebellar Module
# ==========================================

cat("Phase 2 & 3: NSGA-II MoE Search...\n")
# Prepare data for workers
stan_data_val <- list(
  subj = stan_data$subj, RT = stan_data$RT, ITI = stan_data$ITI
)

cl <- makeCluster(cores)
registerDoParallel(cl)
clusterExport(cl, c("stan_data_val", "Switch", "X", "train_subjs", "oob_subjs", "min_RT"))

clusterEvalQ(cl, {
  library(torch)
  torch_set_num_threads(1)
  
  SpatialRNN <- nn_module(
    "SpatialRNN",
    initialize = function(input_dim, hidden_dim, K) {
      self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
      self$policy_head <- nn_linear(hidden_dim, 1)
      self$pi_head <- nn_linear(hidden_dim, K)
      self$mu_head <- nn_linear(hidden_dim, K)
      self$sigma_head <- nn_linear(hidden_dim, K)
      self$tau_head <- nn_linear(hidden_dim, K)
      self$log_var_policy <- nn_parameter(torch_zeros(1))
      self$log_var_kin <- nn_parameter(torch_zeros(1))
      self$K <- K
    },
    forward = function(x, h0 = NULL) {
      out <- self$gru(x, h0)
      h <- out[[1]]
      logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
      logits_pi <- torch_clamp(self$pi_head(h), min=-15, max=15)
      mu_rt <- self$mu_head(h)
      logits_sigma <- torch_clamp(self$sigma_head(h), min=-15, max=15)
      logits_tau <- torch_clamp(self$tau_head(h), min=-15, max=15)
      list(h = h, logits_p = logits_p, logits_pi = logits_pi, mu_rt = mu_rt, logits_sigma = logits_sigma, logits_tau = logits_tau)
    }
  )
  
  MoE_CorticoCerebellar <- nn_module(
    "MoE_CorticoCerebellar",
    initialize = function(rnn_model, alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_granule, beta_granule, gamma_raw) {
      self$rnn <- rnn_model
      self$rnn$requires_grad_(FALSE)
      self$rnn$eval()
      
      # MoE Gate
      self$gamma_raw <- torch_tensor(gamma_raw, dtype=torch_float())
      
      # Biological Hyperparameters (Phase 4 Predictive Coding Array)
      self$alpha_pc <- alpha_pc
      self$lambda_pc <- lambda_pc
      self$beta_thal <- beta_thal
      self$kappa_cf <- kappa_cf
      self$alpha_granule <- alpha_granule
      self$beta_granule <- beta_granule
      
      input_dim <- 27
      hidden_dim <- 32
      granule_dim <- 896
      dcn_dim <- 362
      
      set.seed(42)
      p_granule <- 0.287
      
      W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=32, ncol=27)
      self$W_gen <- torch_tensor(W_gen, dtype=torch_float(), requires_grad=FALSE)
      
      W_thal <- matrix(sample(c(-1, 0, 1), 32 * 362, replace=TRUE, prob=c(p_granule/2, 1-p_granule, p_granule/2)), nrow=362, ncol=32)
      self$W_thal <- torch_tensor(W_thal, dtype=torch_float(), requires_grad=FALSE)
      
      W_fan_in <- matrix(sample(c(-1, 0, 1), 896 * 32, replace=TRUE, prob=c(p_granule/2, 1-p_granule, p_granule/2)), nrow=32, ncol=896)
      self$W_fan_in <- torch_tensor(W_fan_in, dtype=torch_float(), requires_grad=FALSE)
      
      W_DCN <- matrix(sample(c(-1, 0, 1), 362 * 896, replace=TRUE, prob=c(p_granule/2, 1-p_granule, p_granule/2)), nrow=896, ncol=362)
      self$W_DCN <- torch_tensor(W_DCN, dtype=torch_float(), requires_grad=FALSE)
      
      # Readouts
      self$bio_policy_head <- nn_linear(hidden_dim, 1)
      self$bio_pi_head <- nn_linear(hidden_dim, 2)
      self$bio_mu_head <- nn_linear(hidden_dim, 2)
      self$bio_sigma_head <- nn_linear(hidden_dim, 2)
      self$bio_tau_head <- nn_linear(hidden_dim, 2)
      
      self$log_var_policy <- nn_parameter(torch_zeros(1))
      self$log_var_kin <- nn_parameter(torch_zeros(1))
    },
    forward = function(x, iti_seq) {
      rnn_out <- self$rnn(x)
      
      seq_len <- x$size(2)
      device <- x$device
      
      mu_t <- torch_zeros(1, 32, dtype=torch_float(), device=device)
      Z_t <- torch_zeros(1, 896, dtype=torch_float(), device=device)
      D_t <- torch_zeros(1, 362, dtype=torch_float(), device=device)
      Pi <- torch_ones(1, 27, dtype=torch_float(), device=device)
      
      mu_seq <- vector("list", seq_len)
      
      # Biological Euler loop
      for (t in 1:seq_len) {
        I_t <- x[, t, ]
        dt <- iti_seq[t]
        
        I_hat <- torch_matmul(mu_t, self$W_gen)
        epsilon <- Pi * (I_t - I_hat)
        
        err_sum <- torch_sum(torch_abs(epsilon))
        E_CF <- torch_clamp(self$kappa_cf * err_sum, max=1.0)
        
        leak_rate <- torch_clamp(torch_tensor(self$lambda_pc * dt, device=device), max=1.0)
        decay <- leak_rate * mu_t
        
        feedback <- self$beta_thal * torch_matmul(D_t, self$W_thal)
        grad_sensory <- torch_matmul(epsilon, self$W_gen$t())
        
        mu_tilde <- mu_t + self$alpha_pc * (grad_sensory - decay + feedback)
        mu_norm <- torch_sqrt(torch_sum(mu_tilde^2) + 1e-8)
        mu_t <- mu_tilde / torch_clamp(mu_norm, min=1.0)
        
        decay_gran <- torch_clamp(torch_tensor(1.0 - self$beta_granule * dt, device=device), min=0.0)
        Z_t <- Z_t * decay_gran * (1.0 - E_CF) + self$alpha_granule * torch_matmul(mu_t, self$W_fan_in)
        D_t <- torch_matmul(Z_t, self$W_DCN)
        
        mu_seq[[t]] <- mu_t
      }
      
      mu_seq <- torch_stack(mu_seq, dim=2)
      
      bio_logits_p <- torch_clamp(self$bio_policy_head(mu_seq), min=-15, max=15)
      bio_logits_pi <- torch_clamp(self$bio_pi_head(mu_seq), min=-15, max=15)
      bio_mu_rt <- self$bio_mu_head(mu_seq)
      bio_logits_sigma <- torch_clamp(self$bio_sigma_head(mu_seq), min=-15, max=15)
      bio_logits_tau <- torch_clamp(self$bio_tau_head(mu_seq), min=-15, max=15)
      
      gamma <- nnf_sigmoid(self$gamma_raw)
      
      p_total <- (1 - gamma) * rnn_out$logits_p + gamma * bio_logits_p
      pi_total <- (1 - gamma) * rnn_out$logits_pi + gamma * bio_logits_pi
      mu_total <- (1 - gamma) * rnn_out$mu_rt + gamma * bio_mu_rt
      sigma_total <- (1 - gamma) * rnn_out$logits_sigma + gamma * bio_logits_sigma
      tau_total <- (1 - gamma) * rnn_out$logits_tau + gamma * bio_logits_tau
      
      p_switch <- nnf_sigmoid(p_total)
      pi_mix <- nnf_softmax(pi_total, dim = -1)
      sigma_rt <- nnf_softplus(sigma_total) + 1e-4
      tau_rt <- nnf_sigmoid(tau_total) * (0.99 * min_RT)
      
      list(p = p_switch, pi = pi_mix, mu = mu_total, sigma = sigma_rt, tau = tau_rt, gamma = gamma)
    }
  )
})

# 3. Vectorized Fitness Function
evaluate_population <- function(params_matrix) {
  write.table(params_matrix, "moe_checkpoint_pop.csv", append = TRUE, col.names = FALSE, row.names = FALSE, sep=",")
  
  results <- foreach(i = 1:nrow(params_matrix), .combine = rbind, .packages = c("torch")) %dopar% {
    
    alpha_pc <- params_matrix[i, 1]
    lambda_pc <- params_matrix[i, 2]
    beta_thal <- params_matrix[i, 3]
    kappa_cf <- params_matrix[i, 4]
    alpha_granule <- params_matrix[i, 5]
    beta_granule <- params_matrix[i, 6]
    gamma_raw <- params_matrix[i, 7]
    
    rnn_model <- torch_load("frozen_rnn.pt")
    
    model <- MoE_CorticoCerebellar(rnn_model, alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_granule, beta_granule, gamma_raw)
    optimizer <- optim_adam(model$parameters, lr = 0.002, weight_decay = 1e-4)
    
    tryCatch({
      # Train readout weights for 5 epochs
      for (ep in 1:5) {
        model$train()
        for (s in train_subjs) {
          idx <- which(stan_data_val$subj == s)
          x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
          iti_seq <- stan_data_val$ITI[idx]
          y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
          y_rt <- torch_tensor(stan_data_val$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
          
          optimizer$zero_grad()
          preds <- model(x_t, iti_seq)
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
      
      model$eval()
      val_nll_p <- numeric()
      val_nll_r <- numeric()
      with_no_grad({
        for (s in oob_subjs) {
          idx <- which(stan_data_val$subj == s)
          x_t <- torch_tensor(X[idx, ], dtype = torch_float())$unsqueeze(1)
          iti_seq <- stan_data_val$ITI[idx]
          y_switch <- torch_tensor(Switch[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)
          y_rt <- torch_tensor(stan_data_val$RT[idx], dtype = torch_float())$unsqueeze(1)$unsqueeze(3)$expand(c(1, length(idx), 2))
          
          preds <- model(x_t, iti_seq)
          mask <- c(FALSE, rep(TRUE, length(idx)-1))
          
          loss_p <- if (sum(mask) > 0) nnf_binary_cross_entropy(preds$p[, mask, , drop=FALSE], y_switch[, mask, , drop=FALSE])$item() else 0
          
          y_rt_shifted <- y_rt - preds$tau
          log_y <- torch_log(y_rt_shifted + 1e-6)
          log_p_k <- -log_y - 0.5*log(2*pi) - torch_log(preds$sigma) - 0.5 * torch_square((log_y - preds$mu) / preds$sigma)
          log_mix <- torch_logsumexp(torch_log(preds$pi) + log_p_k, dim = -1)
          loss_r <- -log_mix$mean()$item()
          
          val_nll_p <- c(val_nll_p, loss_p)
          val_nll_r <- c(val_nll_r, loss_r)
        }
      })
      
      gamma_val <- as.numeric(nnf_sigmoid(torch_tensor(gamma_raw)))
      total_nll <- mean(val_nll_p) + mean(val_nll_r)
      if (is.nan(total_nll) || is.na(total_nll)) total_nll <- 1000
      
      res_line <- data.frame(alpha_pc=alpha_pc, lambda=lambda_pc, beta_thal=beta_thal, kappa_cf=kappa_cf, 
                             alpha_granule=alpha_granule, beta_granule=beta_granule, gamma_raw=gamma_raw, gamma=gamma_val, NLL=total_nll)
      write.table(res_line, "moe_checkpoint_results.csv", append = TRUE, col.names = FALSE, row.names = FALSE, sep=",")
      
      # Objectives: Minimize NLL, Minimize (1 - gamma)
      c(total_nll, 1.0 - gamma_val)
    }, error = function(e) {
      c(1000, 1.0)
    })
  }
  
  return(t(results))
}

cat("Launching parallel NSGA-II for Phase 3 MoE...\n")
if (file.exists("moe_checkpoint_pop.csv")) file.remove("moe_checkpoint_pop.csv")
if (file.exists("moe_checkpoint_results.csv")) file.remove("moe_checkpoint_results.csv")

# Search space: alpha_pc, lambda, beta_thal, kappa_cf, alpha_granule, beta_granule, gamma_raw
res <- nsga2(evaluate_population, idim = 7, odim = 2, 
             lower.bounds = c(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, -10.0), 
             upper.bounds = c(1.0,   1.0,   1.0,   1.0,   1.0,   1.0,    10.0), 
             popsize = 16, generations = 15, vectorized = TRUE)

stopCluster(cl)
saveRDS(res, 'moe_nsga2_final.rds')
cat("PHASE 3 MOE EVOLUTION COMPLETE.\n")
