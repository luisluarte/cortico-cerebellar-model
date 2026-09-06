library(cmdstanr)
library(torch)
library(jsonlite)
library(loo)

cat("=== Phase 5: Hierarchical Phenotyping (Gamma) ===\n")

# Load Data
stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
mask <- stan_data$subj %in% scaffold_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)
min_RT <- min(stan_data$RT)

# Prep Switch logic & ITI
Switch <- numeric(N_trials)
ITI <- numeric(N_trials)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
Ch <- ifelse(stan_data$Resp == 1, stan_data$Bd1, stan_data$Bd2)

current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0; ITI[i] <- 5.0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
    ITI[i] <- 0.0
  }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# Get Top 1 Model Parameters
res <- readRDS("nsga2_moe_results.rds")
vals <- res$value
pars <- res$par
top1_pars <- pars[order(vals[, 1])[1], ]

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
    logits_p <- torch_clamp(self$policy_head(h), min = -15, max = 15)
    list(
      p = nnf_sigmoid(logits_p),
      pi = nnf_softmax(torch_clamp(self$pi_head(h), min=-15, max=15), dim=-1),
      mu = self$mu_head(h),
      sigma = nnf_softplus(torch_clamp(self$sigma_head(h), min=-15, max=15)) + 1e-4,
      tau = nnf_sigmoid(torch_clamp(self$tau_head(h), min=-15, max=15)) * (0.99 * min_RT)
    )
  }
)
model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()

with_no_grad({
  preds <- model_rnn(torch_tensor(X, dtype=torch_float())$unsqueeze(1))
  p_val <- as.matrix(preds$p$squeeze()); p_val[p_val < 1e-7] <- 1e-7; p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
  rnn_mu <- as.matrix(preds$mu$squeeze())
  rnn_sigma <- as.matrix(preds$sigma$squeeze())
  rnn_tau <- as.matrix(preds$tau$squeeze())
  rnn_pi <- as.matrix(preds$pi$squeeze())
})

# Structural Matrices
set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

# Precompute Noise for Stan
W_purk_noise <- matrix(0, nrow=N_trials, ncol=896)
for(t in 1:N_trials) {
    if(ITI[t] > 0) W_purk_noise[t, ] <- rnorm(896, 0, sqrt(top1_pars[7] * ITI[t]))
}

# Remap Subjects for Stan
subj_map <- as.numeric(factor(stan_data$subj))

stan_data_list <- list(
  N = N_trials,
  S = length(unique(subj_map)),
  subj = subj_map,
  y_switch = Switch,
  y_rt = stan_data$RT,
  ITI = ITI,
  alpha_pc = top1_pars[1],
  lambda_pc = top1_pars[2],
  beta_thal = top1_pars[3],
  kappa_cf = top1_pars[4],
  alpha_granule = top1_pars[5],
  beta_granule = top1_pars[6],
  W_gen = W_gen,
  W_ach1 = W_ach1,
  W_ach2 = W_ach2,
  W_thal = W_thal,
  Pi_vec = Pi_vec,
  X = X,
  W_purk_noise = W_purk_noise,
  rnn_p_logits = as.vector(rnn_p_logits),
  rnn_mu = rnn_mu,
  rnn_sigma = rnn_sigma,
  rnn_tau = rnn_tau,
  rnn_pi = rnn_pi
)

# Fit Model
mod <- cmdstan_model("phenotype_gamma.stan")
fit <- mod$sample(
  data = stan_data_list,
  chains = 8,
  parallel_chains = 8,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100
)

# ELPD Validation via LOO
loo_result <- fit$loo()
cat("\n=== ELPD / LOO-CV Result ===\n")
print(loo_result)

saveRDS(list(fit=fit, loo=loo_result), "phenotype_gamma_results.rds")
cat("Saved to phenotype_gamma_results.rds\n")
