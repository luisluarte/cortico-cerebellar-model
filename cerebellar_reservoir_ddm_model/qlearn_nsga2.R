library(mco)
library(jsonlite)
library(torch)
library(doParallel)

# Setup data and RNN (abbreviated loading similar to previous)
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

# Q-Learning Vectorized Function
q_learning_obj <- function(params_matrix) {
  if (nrow(params_matrix) > ncol(params_matrix)) {
    params_matrix <- t(params_matrix)
  }
  num_ind <- ncol(params_matrix)
  results <- matrix(0, nrow=2, ncol=num_ind)
  
  for (ind in 1:num_ind) {
    alpha_win <- params_matrix[1, ind]
    alpha_loss <- params_matrix[2, ind]
    beta <- params_matrix[3, ind]
    gamma <- params_matrix[4, ind]
    
    Q <- rep(0.5, 8)
    q_logits <- numeric(N_trials)
    
    current_subj <- -1
    for (t in 1:N_trials) {
      if (stan_data$subj[t] != current_subj) {
        Q <- rep(0.5, 8)
        current_subj <- stan_data$subj[t]
      }
      
      b1 <- stan_data$Bd1[t]
      b2 <- stan_data$Bd2[t]
      
      # Probability of choosing Bd1 vs Bd2
      p_bd1 <- plogis(beta * (Q[b1] - Q[b2]))
      
      # Convert to Switch probability
      if (lag_Resp[t] == 1) { # Previously chose Bd1 equivalent (in our context, lag_Resp=1 means chose Bd1 position, wait actually lag_Ch is the exact bandit)
         # If lag_Ch was b1, then p(stay) = p_bd1. p(switch) = 1 - p_bd1
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1
         else p_switch <- p_bd1
      } else {
         if (lag_Ch[t] == b1) p_switch <- 1 - p_bd1
         else p_switch <- p_bd1
      }
      # Clamp
      if (p_switch < 1e-7) p_switch <- 1e-7
      if (p_switch > 1 - 1e-7) p_switch <- 1 - 1e-7
      
      q_logits[t] <- qlogis(p_switch)
      
      # Update Q values (Counterfactual)
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
    
    # Evaluate NLL of blended logits against RNN teacher
    blend_logits <- (1 - gamma) * rnn_p_logits + gamma * q_logits
    # NLL is binary cross entropy against RNN probabilities
    bce <- -mean(p_val * log(plogis(blend_logits)) + (1 - p_val) * log(1 - plogis(blend_logits)))
    
    results[1, ind] <- bce
    results[2, ind] <- 1.0 - gamma
  }
  return(results)
}

# Run NSGA-II
lower_bounds <- c(0.001, 0.001, 0.1, 0.0)
upper_bounds <- c(0.999, 0.999, 10.0, 1.0)

cl <- makeCluster(32)
registerDoParallel(cl)
clusterExport(cl, c("q_learning_obj", "stan_data", "N_trials", "lag_Resp", "lag_Ch", "rnn_p_logits", "p_val"))

res <- nsga2(
  q_learning_obj,
  idim = 4,
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

saveRDS(res, "nsga2_qlearning_results.rds")

# Extract Pareto front metrics
vals <- res$value
pars <- res$par
pareto_idx <- mco::paretoFilter(vals)
# Sort by NLL
sorted_idx <- order(vals[pareto_idx, 1])
p_vals <- vals[pareto_idx, ][sorted_idx, ]
p_pars <- pars[pareto_idx, ][sorted_idx, ]

# Write to CSV
out_df <- data.frame(
  NLL = p_vals[, 1],
  RNN_Reliance = p_vals[, 2],
  Gamma = p_pars[, 4],
  Alpha_Win = p_pars[, 1],
  Alpha_Loss = p_pars[, 2],
  Beta = p_pars[, 3]
)
write.csv(out_df, "qlearning_pareto.csv", row.names=FALSE)
cat("Q-Learning Pareto calculation complete. Saved to qlearning_pareto.csv\n")
