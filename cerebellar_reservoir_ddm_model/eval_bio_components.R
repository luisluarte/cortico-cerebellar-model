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

SpatialRNN <- nn_module("SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
    self$mu_head <- nn_linear(hidden_dim, K)
  },
  forward = function(x, h0 = NULL) {
    h <- self$gru(x, h0)[[1]]
    list(p = nnf_sigmoid(torch_clamp(self$policy_head(h), min = -15, max = 15)), mu = self$mu_head(h))
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
})

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
ITI <- c(5.0, rep(0.0, N_trials - 1))

for (t in 1:N_trials) {
  if (stan_data$subj[t] != current_subj) {
    ITI[t] <- 5.0
    current_subj <- stan_data$subj[t]
  }
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

XX_inv <- solve(crossprod(mu_history) + diag(0.01, 32))
W_policy <- XX_inv %*% crossprod(mu_history, rnn_p_logits)
W_mu <- XX_inv %*% crossprod(mu_history, rnn_mu)

bio_logits <- mu_history %*% W_policy
bio_mu <- mu_history %*% W_mu

gamma <- top1_pars[8]
blend_logits <- (1 - gamma) * rnn_p_logits + gamma * bio_logits
blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu

bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
rmse <- sqrt(mean((rnn_mu - blend_mu)^2))

cat(sprintf("Bio BCE: %.4f\n", bce))
cat(sprintf("Bio RMSE: %.4f\n", rmse))
cat(sprintf("Bio Total (BCE + RMSE): %.4f\n", bce + rmse))

# Autonomous Bio Total (gamma = 1)
auton_bce <- -mean(p_val * log(plogis(bio_logits)) + (1 - p_val) * log(1 - plogis(bio_logits)))
auton_rmse <- sqrt(mean((rnn_mu - bio_mu)^2))
cat(sprintf("Autonomous Bio BCE: %.4f\n", auton_bce))
cat(sprintf("Autonomous Bio RMSE: %.4f\n", auton_rmse))
cat(sprintf("Autonomous Bio Total: %.4f\n", auton_bce + auton_rmse))
