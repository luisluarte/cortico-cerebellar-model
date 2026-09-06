
library(jsonlite)
library(torch)

# Load Data
stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)

# Select one subject for the visual
target_subj <- scaffold_subjs[1]
mask <- stan_data$subj == target_subj
sub_data <- stan_data[mask, ]
N_trials <- nrow(sub_data)

# Prep Inputs
Ch <- ifelse(sub_data$Resp == 1, sub_data$Bd1, sub_data$Bd2)
lag_Reward <- c(0, sub_data$Reward[-N_trials])
lag_Ch <- c(0, Ch[-N_trials])
lag_RT <- c(0, sub_data$RT[-N_trials])
lag_Resp <- c(0, ifelse(sub_data$Resp[-N_trials] == 1, 1, 0))

Switch <- c(0, ifelse(sub_data$Resp[-1] != sub_data$Resp[-N_trials], 1, 0))
ITI <- c(5.0, rep(0.0, N_trials - 1))

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, sub_data$Bd1[i]] <- 1; X_Bd2[i, sub_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# Get RNN Predictions
SpatialRNN <- nn_module(
  "SpatialRNN",
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
    list(p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)))
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

with_no_grad({
  preds <- model_rnn(torch_tensor(X, dtype=torch_float())$unsqueeze(1))
  rnn_p <- as.numeric(preds$p$squeeze())
})

# Get Biological Predictions (using Top 1 Model parameters)
res <- readRDS("nsga2_moe_results.rds")
top1_pars <- res$par[order(res$value[, 1])[1], ]

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
mu_history <- matrix(0, nrow=N_trials, ncol=32)

for (t in 1:N_trials) {
  if (ITI[t] > 0) {
    Z <- rep(0, 896)
    W_purk <- W_purk + rnorm(896, 0, sqrt(top1_pars[7] * ITI[t]))
  }
  I_t <- X[t, ]
  I_hat <- as.numeric(W_gen %*% mu)
  eps <- Pi_vec * (I_t - I_hat)
  mu <- mu + top1_pars[1] * (as.numeric(t(W_gen) %*% eps) - (top1_pars[2] * ITI[t]) * mu + top1_pars[3] * as.numeric(W_thal %*% D))
  G <- as.numeric(W_ach1 %*% mu)
  Z <- (1.0 - top1_pars[6]) * Z + top1_pars[5] * G
  eps_mag <- mean(abs(eps))
  W_purk <- W_purk - top1_pars[4] * (eps_mag * Z)
  D <- as.numeric(W_ach2 %*% (G * W_purk))
  mu_history[t, ] <- mu
}

# Dummy Ridge readout mapping for visual proxy (usually trained on full pop, but for visual approximation we regress on this subject)
XX <- crossprod(mu_history) + diag(0.01, 32)
W_policy <- solve(XX) %*% crossprod(mu_history, qlogis(rnn_p + 1e-7))
bio_logits <- as.numeric(mu_history %*% W_policy)
bio_p <- plogis(bio_logits)

# Plotting
df <- data.frame(
  Trial = 1:60,  # Zoom in on first 60 trials
  Human = Switch[1:60],
  RNN = rnn_p[1:60],
  Bio = bio_p[1:60]
)

write.csv(df, "denoising_data.csv", row.names=FALSE)
cat("Data saved to denoising_data.csv\n")
