library(torch)
library(jsonlite)
library(pROC)
library(mco)
library(doParallel)

# 1. Initialization and Data Loading
cat("=== Initializing Phase 2/3 MoE NSGA-II Pipeline ===\n")
device <- torch_device("cpu")
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

# We evaluate Phase 2/3 strictly on the UNSEEN scaffold subjects
scaffold_subjs <- setdiff(unique(stan_data$subj), rnn_phase1_subjs)
mask <- stan_data$subj %in% scaffold_subjs
stan_data$subj <- stan_data$subj[mask]
stan_data$Resp <- stan_data$Resp[mask]
stan_data$Bd1 <- stan_data$Bd1[mask]
stan_data$Bd2 <- stan_data$Bd2[mask]
stan_data$Reward <- stan_data$Reward[mask]
stan_data$RT <- stan_data$RT[mask]
N_trials <- sum(mask)
min_RT <- min(stan_data$RT)

Ch <- numeric(N_trials)
for(i in 1:N_trials) Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])

Switch <- numeric(N_trials); lag_Reward <- numeric(N_trials); lag_Ch <- numeric(N_trials); lag_RT <- numeric(N_trials); lag_Resp <- numeric(N_trials)
ITI <- numeric(N_trials) # Simulate ITI boundaries at the start of each subject's block
current_subj <- -1
for(i in 1:N_trials) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0; ITI[i] <- 5.0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(stan_data$Resp[i] != stan_data$Resp[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
    ITI[i] <- 0.0 # 0 denotes continuous within-block sequence
  }
}

X_Bd1 <- matrix(0, nrow = N_trials, ncol = 8); X_Bd2 <- matrix(0, nrow = N_trials, ncol = 8); X_lag_Ch <- matrix(0, nrow = N_trials, ncol = 8)
for (i in 1:N_trials) { X_Bd1[i, stan_data$Bd1[i]] <- 1; X_Bd2[i, stan_data$Bd2[i]] <- 1; if (lag_Ch[i] > 0) X_lag_Ch[i, lag_Ch[i]] <- 1 }
X <- cbind(X_Bd1, X_Bd2, lag_Reward, lag_RT, X_lag_Ch, lag_Resp)

# 2. Precompute RNN Asymptotic Ceilings (Structural Firewall)
cat("Precomputing Deterministic RNN Ceilings...\n")
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
    pi_mix <- nnf_softmax(torch_clamp(self$pi_head(h), min=-15, max=15), dim=-1)
    mu_rt <- self$mu_head(h)
    sigma_rt <- nnf_softplus(torch_clamp(self$sigma_head(h), min=-15, max=15)) + 1e-4
    tau_rt <- nnf_sigmoid(torch_clamp(self$tau_head(h), min=-15, max=15)) * (0.99 * min_RT)
    list(p = p_switch, pi = pi_mix, mu = mu_rt, sigma = sigma_rt, tau = tau_rt)
  }
)

model_rnn <- SpatialRNN(ncol(X), 4, 2)
model_rnn$load_state_dict(torch_load("frozen_rnn_baseline.pt"))
model_rnn$eval()
x_t <- torch_tensor(X, dtype = torch_float())$unsqueeze(1)
with_no_grad({
  preds <- model_rnn(x_t)
  
  # Inverse sigmoid (qlogis) to recover logits without breaking encapsulation
  p_val <- as.matrix(preds$p$squeeze())
  p_val[p_val < 1e-7] <- 1e-7
  p_val[p_val > 1 - 1e-7] <- 1 - 1e-7
  rnn_p_logits <- qlogis(p_val)
  
  rnn_mu <- as.matrix(preds$mu$squeeze())
  rnn_sigma <- as.matrix(preds$sigma$squeeze())
  rnn_tau <- as.matrix(preds$tau$squeeze())
  rnn_pi <- as.matrix(preds$pi$squeeze())
})

# 3. Biological Reservoir Setup (Base R for massive CPU speedup)
set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)
Y_switch <- Switch
Y_RT <- stan_data$RT

# Enable parallel cores
num_cores <- parallel::detectCores()
registerDoParallel(cores = num_cores)
cat(sprintf("Saturating all %d available cores for NSGA-II search...\n", num_cores))

# 4. Objective Function (Fully Base R optimized, zero torch backprop)
eval_population <- function(pop_matrix) {
  results <- foreach(i = 1:nrow(pop_matrix), .combine=rbind) %dopar% {
    theta <- pop_matrix[i, ]
    alpha_pc      <- theta[1]
    lambda_pc     <- theta[2]
    beta_thal     <- theta[3]
    kappa_cf      <- theta[4]
    alpha_granule <- theta[5]
    beta_granule  <- theta[6]
    sigma2_diff   <- theta[7]
    gamma_raw     <- theta[8]
    
    mu_history <- matrix(0, nrow=N_trials, ncol=32)
    mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
    
    # Fast Euler Loop
    for (t in 1:N_trials) {
      if (ITI[t] > 0) {
        Z[] <- 0
        W_purk <- W_purk + rnorm(896, 0, sqrt(sigma2_diff * ITI[t]))
      }
      
      I_t <- X[t, ]
      I_hat <- as.numeric(W_gen %*% mu)
      eps <- Pi_vec * (I_t - I_hat)
      
      mu <- mu + alpha_pc * (as.numeric(t(W_gen) %*% eps) - (lambda_pc * ITI[t]) * mu + beta_thal * as.numeric(W_thal %*% D))
      G <- as.numeric(W_ach1 %*% mu)
      Z <- (1 - beta_granule) * Z + alpha_granule * G
      
      eps_mag <- mean(abs(eps))
      W_purk <- W_purk - kappa_cf * (eps_mag * Z)
      D <- as.numeric(W_ach2 %*% (G * W_purk))
      
      mu_history[t, ] <- mu
    }
    
    # Divergence Check
    if (any(is.na(mu_history)) || any(is.infinite(mu_history))) {
      return(c(1e6, 1e6))
    }
    
    # Analytic Ridge Regression to find optimal Bio Projections (Matching RNN Targets)
    lambda_ridge <- 0.01
    XX_inv <- tryCatch({
      solve(crossprod(mu_history) + diag(lambda_ridge, 32))
    }, error = function(e) { NULL })
    
    if (is.null(XX_inv)) {
      return(c(1e6, 1e6)) # Penalty for computationally singular / exploding state
    }
    
    bio_policy_logits <- mu_history %*% (XX_inv %*% crossprod(mu_history, rnn_p_logits))
    bio_mu <- mu_history %*% (XX_inv %*% crossprod(mu_history, rnn_mu))
    
    # Mixture of Experts Projection
    gamma <- 1 / (1 + exp(-gamma_raw))
    
    blend_policy_logits <- (1 - gamma) * rnn_p_logits + gamma * bio_policy_logits
    blend_p <- 1 / (1 + exp(-blend_policy_logits))
    blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu
    
    # NLL Calculation (Vectorized)
    bce <- -(Y_switch * log(blend_p + 1e-7) + (1 - Y_switch) * log(1 - blend_p + 1e-7))
    loss_p <- mean(bce)
    
    # Kinematic Mixture NLL
    log_y_tau <- log(Y_RT - rnn_tau + 1e-6)
    log_p_k1 <- -log_y_tau[,1] - 0.5*log(2*pi) - log(rnn_sigma[,1]) - 0.5 * ((log_y_tau[,1] - blend_mu[,1]) / rnn_sigma[,1])^2
    log_p_k2 <- -log_y_tau[,2] - 0.5*log(2*pi) - log(rnn_sigma[,2]) - 0.5 * ((log_y_tau[,2] - blend_mu[,2]) / rnn_sigma[,2])^2
    
    max_log <- pmax(log(rnn_pi[,1]) + log_p_k1, log(rnn_pi[,2]) + log_p_k2)
    sum_exp <- exp(log(rnn_pi[,1]) + log_p_k1 - max_log) + exp(log(rnn_pi[,2]) + log_p_k2 - max_log)
    log_mix <- max_log + log(sum_exp)
    loss_r <- mean(-log_mix, na.rm=TRUE)
    
    if (is.na(loss_r) || is.infinite(loss_r)) loss_r <- 1000
    if (is.na(loss_p) || is.infinite(loss_p)) loss_p <- 1000
    
    NLL <- loss_p + loss_r
    f2 <- 1 - gamma
    
    c(NLL, f2)
  }
  return(t(results))
}

# 5. NSGA-II Evolution
cat("Starting NSGA-II Evolutionary Search...\n")
lower_bounds <- c(0.001, 0.001, 0.001, 0.0001, 0.01, 0.01, 0.0001, -5.0)
upper_bounds <- c(0.5,   0.5,   1.0,   0.1,    1.0,  1.0,  0.1,     5.0)

results_nsga <- nsga2(
  fn = eval_population,
  idim = 8,
  odim = 2,
  vectorized = TRUE,
  popsize = num_cores * 2, # Scale population to cores 
  generations = 50,
  lower.bounds = lower_bounds,
  upper.bounds = upper_bounds
)

saveRDS(results_nsga, "nsga2_moe_results.rds")
cat("=== Evolution Complete! Pareto Front Saved. ===\n")
