local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
    library(torch)
    library(jsonlite)
    library(dplyr)
})

cat("Loading data...\n")
stan_data <- read_json("../../data/stan_data.json", simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

# Ensure CUDA is available like in run_ece.py
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("running on device: %s\n", as.character(device)))

N_trials <- stan_data$N
min_RT <- min(stan_data$RT)

Ch <- numeric(N_trials)
for (i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials)
lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)

current_subj <- -1
for (i in 1:N_trials) {
    if (stan_data$subj[i] != current_subj) {
        Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0
        current_subj <- stan_data$subj[i]
    } else {
        Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
        lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
    }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8)
X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8)
X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) {
    X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1
    if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1
}
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# Model definition
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
        p_switch <- nnf_sigmoid(logits_p)
        pi_mix <- nnf_softmax(torch_clamp(self$pi_head(h), min = -15, max = 15), dim = -1)
        mu_rt <- self$mu_head(h)
        sigma_rt <- nnf_softplus(torch_clamp(self$sigma_head(h), min = -15, max = 15)) + 1e-4
        tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min = -15, max = 15)) * (0.99 * min_RT)
        list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
    }
)

cat("Loading frozen model...\n")
model <- torch_load("../../results/frozen_rnn_baseline.pt", device = "cpu")
model <- model$to(device = device)
model$eval()

all_logits <- numeric()
all_mu <- numeric()
all_sigma <- numeric()
all_labels <- numeric()
all_lag_reward <- numeric()
all_lag_ch <- numeric()
all_lag_resp <- numeric()
all_subjs <- numeric()
all_X <- matrix(nrow = 0, ncol = ncol(X))
all_ITI <- numeric()

cat("Computing precise distribution targets...\n")
with_no_grad({
    for (s in rnn_phase1_subjs) {
        idx <- which(stan_data$subj == s)
        x_t <- torch_tensor(X[idx, ], dtype=torch_float(), device=device)$unsqueeze(1)
        
        preds <- model(x_t)
        
        mask <- c(FALSE, rep(TRUE, length(idx)-1))
        
        # Get elements one by one, keeping them alive
        p_t <- as.numeric(preds$p$cpu())
        # Convert sigmoid probabilities back to raw logits for Platt Scaling
        logits_t <- qlogis(pmin(pmax(p_t, 1e-7), 1 - 1e-7))
        
        mu_t_raw <- as.numeric(preds$mu$cpu())
        mu_t <- matrix(mu_t_raw, nrow = length(idx), byrow=TRUE)
        
        sigma_t_raw <- as.numeric(preds$sigma$cpu())
        sigma_t <- matrix(sigma_t_raw, nrow = length(idx), byrow=TRUE)
        
        y_t <- Switch[idx]
        
        all_logits <- c(all_logits, logits_t[mask])
        
        all_mu <- c(all_mu, as.numeric(t(mu_t[mask, , drop=FALSE])))
        all_sigma <- c(all_sigma, as.numeric(t(sigma_t[mask, , drop=FALSE])))
        
        all_labels <- c(all_labels, y_t[mask])
        
        all_lag_reward <- c(all_lag_reward, lag_Reward[idx][mask])
        all_lag_ch <- c(all_lag_ch, lag_Ch[idx][mask])
        all_lag_resp <- c(all_lag_resp, lag_Resp[idx][mask])
        all_subjs <- c(all_subjs, rep(s, sum(mask)))
        all_X <- rbind(all_X, X[idx, , drop=FALSE][mask, , drop=FALSE])
        all_ITI <- c(all_ITI, c(5.0, rep(0.0, sum(mask)-1)))
    }
})

num_valid <- length(all_logits)
mu_matrix <- matrix(all_mu, nrow = num_valid, byrow=TRUE)
sigma_matrix <- matrix(all_sigma, nrow = num_valid, byrow=TRUE)

# Map 2-D mixture into a 32-D target vector by duplicating columns
rnn_mu_32 <- matrix(rep(mu_matrix, 16), ncol = 32)
rnn_sigma_32 <- matrix(rep(sigma_matrix, 16), ncol = 32)

targets <- list(
    rnn_logits = all_logits,
    rnn_mu = rnn_mu_32,
    rnn_sigma = rnn_sigma_32,
    labels = all_labels,
    lag_reward = all_lag_reward,
    lag_ch = all_lag_ch,
    lag_resp = all_lag_resp,
    subjs = all_subjs,
    X = all_X,
    ITI = all_ITI
)

saveRDS(targets, "distillation_targets_v2.rds")
cat("Saved to distillation_targets_v2.rds\n")
