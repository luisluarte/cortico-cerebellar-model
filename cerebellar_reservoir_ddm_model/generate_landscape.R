# setup
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) dir.create(local_lib, recursive = TRUE)
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!require("pacman", character.only = TRUE)) install.packages("pacman", lib = local_lib)
pacman::p_load(torch, jsonlite, dplyr)

cat("Loading data...\n")
stan_data <- read_json("../data/stan_data_N100.json", simplifyVector = TRUE)
N_trials_total <- stan_data$N

# 1. Filter to just 10 subjects
test_subjs <- unique(stan_data$subj)[1:10]
idx_10 <- which(stan_data$subj %in% test_subjs)

stan_data_sub <- list(
  subj = stan_data$subj[idx_10],
  Resp = stan_data$Resp[idx_10],
  Reward = stan_data$Reward[idx_10],
  RT = stan_data$RT[idx_10],
  Bd1 = stan_data$Bd1[idx_10],
  Bd2 = stan_data$Bd2[idx_10],
  ITI = stan_data$ITI[idx_10]
)
N_trials <- length(idx_10)

# Process features
Ch <- numeric(N_trials_total)
for(i in 1:N_trials_total) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch_total <- numeric(N_trials_total)
lag_Reward_total <- numeric(N_trials_total)
lag_Ch_total <- numeric(N_trials_total)
lag_RT_total <- numeric(N_trials_total)
lag_Resp_total <- numeric(N_trials_total)

current_subj <- -1
for(i in 1:N_trials_total) {
  if (stan_data$subj[i] != current_subj) {
    Switch_total[i] <- 0; lag_Reward_total[i] <- 0; lag_Ch_total[i] <- 0; lag_RT_total[i] <- 0; lag_Resp_total[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch_total[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
    lag_Reward_total[i] <- stan_data$Reward[i-1]
    lag_Ch_total[i] <- Ch[i-1]
    lag_RT_total[i] <- stan_data$RT[i-1]
    lag_Resp_total[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
  }
}

Switch <- Switch_total[idx_10]
lag_Reward <- lag_Reward_total[idx_10]
lag_Ch <- lag_Ch_total[idx_10]
lag_RT <- lag_RT_total[idx_10]
lag_Resp <- lag_Resp_total[idx_10]
ITI <- stan_data_sub$ITI

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8)
X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8)
X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { 
  X_Bd1[i, stan_data_sub$Bd1[i]] <- 1
  X_Bd2[i, stan_data_sub$Bd2[i]] <- 1
  if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 
}
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

device <- torch_device("cpu")

# Define Model (to load Oracle)
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
  },
  forward = function(x, h0 = NULL) {
    h <- self$gru(x, h0)[[1]]
    list(
      p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)),
      mu = self$mu_head(h)
    )
  }
)

cat("Loading RNN Oracle...\n")
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt", device = "cpu"))
model_rnn$eval()

x_t <- torch_tensor(X, dtype = torch_float(), device = device)$unsqueeze(1)
with_no_grad({
  preds <- model_rnn(x_t)
  p_val <- as.numeric(preds$p$squeeze())
})
rnn_p_logits <- qlogis(p_val)

# Random fixed projection matrices for cerebellum
set.seed(123)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(27)), nrow = 27, ncol = 32)
W_ach1 <- t(matrix(rnorm(32 * 896, 0, 1/sqrt(32)), nrow = 32, ncol = 896))
W_ach2 <- t(matrix(rnorm(896 * 362, 0, 1/sqrt(896)), nrow = 896, ncol = 362))
W_thal <- t(matrix(rnorm(362 * 32, 0, 1/sqrt(362)), nrow = 362, ncol = 32))
Pi_vec <- rbeta(27, 2, 5)

# Fixed parameters for the other dimensions
p_beta_thal <- 0.2
p_alpha_gran <- 0.5
p_beta_gran <- 0.5
p_sigma2_diff <- 0.1

# Grid setup
grid_res <- 100
alpha_pc_seq <- seq(0.0001, 0.002, length.out = grid_res)
kappa_cf_seq <- seq(0.0001, 0.05, length.out = grid_res)
p_lambda_pc <- 0.2 # fixed leak

results_df <- data.frame(
  alpha_pc = numeric(),
  kappa_cf = numeric(),
  empirical_nll = numeric(),
  rnn_nll = numeric()
)

cat("Computing Loss Landscapes (Grid Size:", grid_res, "x", grid_res, ")...\n")
count <- 1
for (a in alpha_pc_seq) {
  for (k in kappa_cf_seq) {
    
    mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
    mu_history <- matrix(0, nrow=N_trials, ncol=32)
    diverged <- FALSE
    
    for (t in 1:N_trials) {
      if (ITI[t] > 0) {
        Z <- rep(0, 896)
        W_purk <- W_purk + rnorm(896, 0, sqrt(p_sigma2_diff * ITI[t]))
      }
      I_t <- X[t, ]
      I_hat <- as.numeric(W_gen %*% mu)
      eps <- Pi_vec * (I_t - I_hat)
      mu <- mu + a * (as.numeric(t(W_gen) %*% eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * as.numeric(W_thal %*% D))
      
      if (any(is.na(mu)) || max(abs(mu)) > 1e4) { diverged <- TRUE; break }
      
      G <- as.numeric(W_ach1 %*% mu)
      Z <- (1.0 - p_beta_gran) * Z + p_alpha_gran * G
      eps_mag <- mean(abs(eps))
      W_purk <- W_purk - k * (eps_mag * Z)
      D <- as.numeric(W_ach2 %*% (G * W_purk))
      mu_history[t, ] <- mu
    }
    
    if (diverged) {
      emp_loss <- 10.0
      rnn_loss <- 10.0
    } else {
      # Use a larger ridge penalty to ensure stability
      XX_inv <- tryCatch(solve(crossprod(mu_history) + diag(1.0, 32)), error = function(e) NULL)
      if (is.null(XX_inv)) {
        emp_loss <- 10.0; rnn_loss <- 10.0
      } else {
        # Fit W_policy to RNN logits (this perfectly distills the knowledge into the linear readout)
        W_policy <- XX_inv %*% crossprod(mu_history, rnn_p_logits)
        bio_logits <- as.numeric(mu_history %*% W_policy)
        
        # Clip logits to prevent NaNs in plogis
        bio_logits <- pmax(pmin(bio_logits, 15), -15)
        
        # 1. Empirical Loss (Hard Targets: Switch)
        emp_loss <- -mean(Switch * log(plogis(bio_logits) + 1e-7) + (1 - Switch) * log(1 - plogis(bio_logits) + 1e-7))
        
        # 2. Distilled Loss (Soft Targets: p_val from RNN)
        rnn_loss <- -mean(p_val * log(plogis(bio_logits) + 1e-7) + (1 - p_val) * log(1 - plogis(bio_logits) + 1e-7))
      }
    }
    
    results_df <- rbind(results_df, data.frame(
      alpha_pc = a,
      kappa_cf = k,
      empirical_nll = emp_loss,
      rnn_nll = rnn_loss
    ))
    count <- count + 1
  }
}

saveRDS(results_df, "landscape_data.rds")
cat("Finished! Saved to landscape_data.rds\n")
