library(mco)
library(jsonlite)
library(torch)
library(doParallel)

# Setup data and RNN
stan_data <- as.data.frame(read_json('data/stan_data_N100.json', simplifyVector = TRUE))
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
mask <- stan_data$subj %in% scaffold_subjs
stan_data <- stan_data[mask, ]
N_trials <- nrow(stan_data)

Switch <- numeric(N_trials)
Ch <- ifelse(stan_data$Resp == 1, stan_data$Bd1, stan_data$Bd2)
lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
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

SpatialRNN <- nn_module(
  "SpatialRNN",
  initialize = function(input_dim, hidden_dim, K) {
    self$gru <- nn_gru(input_dim, hidden_dim, num_layers = 1, batch_first = TRUE)
    self$policy_head <- nn_linear(hidden_dim, 1)
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
  p_val <- as.numeric(preds$p$squeeze())
  p_val[p_val < 1e-7] <- 1e-7; p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
})

# Parametrized WSLS Objective
wsls_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) {
    params_matrix <- t(params_matrix)
  }
  num_ind <- ncol(params_matrix)
  results <- matrix(0, nrow=2, ncol=num_ind)
  
  for (ind in 1:num_ind) {
    theta_win <- params_matrix[1, ind]
    theta_loss <- params_matrix[2, ind]
    gamma <- params_matrix[3, ind]
    
    # 1e-7 Clamping implemented inherently by bounds, but enforce safely
    theta_win <- max(1e-7, min(1 - 1e-7, theta_win))
    theta_loss <- max(1e-7, min(1 - 1e-7, theta_loss))
    
    # Vectorized WSLS prediction
    wsls_p <- ifelse(lag_Reward > 0, theta_win, theta_loss)
    wsls_p[lag_Reward == 0 & lag_Ch == 0] <- 0.5 # Neutral start
    
    # Blend in logit space
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * qlogis(wsls_p)
    
    # Evaluate BCE against RNN Teacher
    bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
    
    results[1, ind] <- bce
    results[2, ind] <- 1.0 - gamma
  }
  return(results)
}

# Run NSGA-II (theta_win, theta_loss, gamma)
lower_bounds <- c(0.001, 0.001, 0.0)
upper_bounds <- c(0.999, 0.999, 1.0)

cl <- makeCluster(16)
registerDoParallel(cl)
clusterExport(cl, c("wsls_obj", "stan_data", "N_trials", "lag_Reward", "lag_Ch", "rnn_p_logits", "p_val"))

res <- nsga2(
  wsls_obj,
  idim = 3,
  odim = 2,
  lower.bounds = lower_bounds,
  upper.bounds = upper_bounds,
  popsize = 100,
  generations = 30,
  cprob = 0.9,
  mprob = 0.2,
  vectorized = TRUE
)
stopCluster(cl)

saveRDS(res, "nsga2_wsls_results.rds")
cat("WSLS Pareto calculation complete.\n")
