
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
library(jsonlite)
library(loo)

cat("Loading Joint Fit...\n")
fit <- readRDS("joint_fit.rds")

stan_data <- read_json("../../data/stan_data_N100_distilled.json", simplifyVector = TRUE)
train_subjs <- readRDS("rnn_phase1_subjects.rds")
N <- stan_data$N

Ch <- numeric(N)
Choice_binary <- numeric(N)
for(i in 1:N) {
  Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])
  Choice_binary[i] <- ifelse(stan_data$Resp[i] == 1, 1, 0)
}

Switch <- numeric(N)
lag_Reward <- numeric(N)
lag_Ch <- numeric(N)
lag_RT <- numeric(N)
lag_Resp <- numeric(N)

current_subj <- -1
for(i in 1:N) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0 
    lag_Reward[i] <- 0
    lag_Ch[i] <- 0
    lag_RT[i] <- 0
    lag_Resp[i] <- 0
    current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]
    lag_Ch[i] <- Ch[i-1]
    lag_RT[i] <- stan_data$RT[i-1]
    lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
  }
}

X <- matrix(0, nrow = N, ncol = 27)
for (i in 1:N) {
  X[i, stan_data$Bd1[i]] <- 1
  X[i, stan_data$Bd2[i] + 8] <- 1
  X[i, 17] <- lag_Reward[i]
  X[i, 18] <- lag_RT[i]
  if (lag_Ch[i] > 0) { X[i, 18 + lag_Ch[i]] <- 1 }
  X[i, 27] <- lag_Resp[i]
}

set.seed(42)
selected_5_indist <- sample(train_subjs, 5)
valid_idx <- which(lag_Resp >= 0 & stan_data$subj %in% selected_5_indist & lag_Ch > 0)

X_final <- X[valid_idx, ]
y_switch <- Switch[valid_idx]
subj_id <- as.numeric(as.factor(stan_data$subj[valid_idx]))
iti <- stan_data$ITI[valid_idx]
N_final <- length(valid_idx)

Pi_vec <- rep(1, 27)

W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)

draws <- fit$draws()
S <- 1000

get_vec <- function(name, len) {
  sapply(1:len, function(i) draws[, paste0(name, "[", i, "]")])
}
get_mat <- function(name, rows, cols) {
  arr <- array(0, dim=c(S, rows, cols))
  for(r in 1:rows) {
    for(c in 1:cols) {
      arr[, r, c] <- draws[, paste0(name, "[", r, ",", c, "]")]
    }
  }
  arr
}

alpha_pc <- get_vec("alpha_pc", 5)
lambda_pc <- get_vec("lambda_pc", 5)
beta_thal <- get_vec("beta_thal", 5)
kappa_cf <- get_vec("kappa_cf", 5)
alpha_gran <- get_vec("alpha_gran", 5)
beta_gran <- get_vec("beta_gran", 5)
intercept_c <- get_vec("intercept_c", 5)
W_c <- get_mat("W_c", 32, 5)

cat("Simulating Traces...\n")
ll_matrix <- matrix(0, nrow=S, ncol=N_final)

for(s in 1:S) {
  mu <- rep(0, 32)
  Z <- rep(0, 896)
  Wp <- rep(0, 896)
  D <- rep(0, 362)
  
  for(t in 1:N_final) {
    k <- subj_id[t]
    if(t > 1 && subj_id[t] != subj_id[t-1]) {
      mu <- rep(0, 32); Z <- rep(0, 896); Wp <- rep(0, 896); D <- rep(0, 362)
    }
    
    I_t <- X_final[t, ]
    I_hat <- W_gen %*% mu
    eps <- Pi_vec * (I_t - I_hat)
    
    U <- (t(W_gen) %*% eps) - (lambda_pc[s, k] * iti[t]) * mu + beta_thal[s, k] * (W_thal %*% D)
    mu <- mu + alpha_pc[s, k] * U
    
    G <- W_ach1 %*% mu
    Z <- (1.0 - beta_gran[s, k]) * Z + alpha_gran[s, k] * G
    
    eps_mag <- mean(abs(eps))
    Wp <- Wp - kappa_cf[s, k] * (eps_mag * Z)
    
    D <- W_ach2 %*% (G * Wp)
    
    logit <- intercept_c[s, k] + sum(mu * W_c[s, , k])
    prob <- 1 / (1 + exp(-logit))
    
    if(y_switch[t] == 1) {
      ll_matrix[s, t] <- log(prob + 1e-12)
    } else {
      ll_matrix[s, t] <- log(1 - prob + 1e-12)
    }
  }
}

cat("NAs in ll_matrix:", sum(is.na(ll_matrix)), "\n")
cat("NaNs in ll_matrix:", sum(is.nan(ll_matrix)), "\n")
cat("Infs in ll_matrix:", sum(is.infinite(ll_matrix)), "\n")
ll_matrix[is.na(ll_matrix) | is.nan(ll_matrix) | is.infinite(ll_matrix)] <- log(1e-12)
cat("Computing LOO...\n")
loo_joint <- loo(ll_matrix)
print(loo_joint)
