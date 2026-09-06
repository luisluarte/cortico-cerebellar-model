library(jsonlite)
library(torch)

stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
mask <- stan_data$subj %in% scaffold_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)

Ch <- ifelse(stan_data$Resp == 1, stan_data$Bd1, stan_data$Bd2)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
  }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

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
    list(p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)),
         mu = self$mu_head(h))
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

with_no_grad({
  preds <- model_rnn(torch_tensor(X, dtype=torch_float())$unsqueeze(1))
  rnn_mu <- as.matrix(preds$mu$squeeze())
})

# Q-Learning Trace Generation
alpha_win <- 0.9999
alpha_loss <- 0.6892
beta <- 0.9181

Q <- rep(0.5, 8)
Q_history <- matrix(0, nrow=N_trials, ncol=8)
current_subj <- -1

for (t in 1:N_trials) {
  if (stan_data$subj[t] != current_subj) {
    Q <- rep(0.5, 8)
    current_subj <- stan_data$subj[t]
  }
  Q_history[t, ] <- Q
  b1 <- stan_data$Bd1[t]
  b2 <- stan_data$Bd2[t]
  chosen <- ifelse(stan_data$Resp[t] == 1, b1, b2)
  unchosen <- ifelse(stan_data$Resp[t] == 1, b2, b1)
  r <- stan_data$Reward[t]
  if (r > 0) {
     Q[chosen] <- Q[chosen] + alpha_win * (1 - Q[chosen])
     Q[unchosen] <- Q[unchosen] + alpha_win * (0 - Q[unchosen])
  } else {
     Q[chosen] <- Q[chosen] + alpha_loss * (0 - Q[chosen])
     Q[unchosen] <- Q[unchosen] + alpha_loss * (1 - Q[unchosen])
  }
}

# Q-Learning RT Mapping (Ridge Regression from Q-values to rnn_mu)
XX_inv <- solve(crossprod(Q_history) + diag(0.01, 8))
W_mu <- XX_inv %*% crossprod(Q_history, rnn_mu)
q_mu <- Q_history %*% W_mu

rmse_mu <- sqrt(mean((rnn_mu - q_mu)^2))

cat(sprintf("Q-Learning RT RMSE (against RNN teacher): %.4f\n", rmse_mu))
cat(sprintf("Q-Learning Total NLL (Optimal BCE + RMSE): %.4f\n", 0.6896 + rmse_mu))
