
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(jsonlite)
})

cat("Loading Data...\n")
stan_data <- read_json("../../data/stan_data_N100_distilled.json", simplifyVector = TRUE)
train_subjs <- readRDS("rnn_phase1_subjects.rds")
N <- stan_data$N

Ch <- numeric(N)
for(i in 1:N) {
  Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i])
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
selected_10_indist <- sample(train_subjs, 10)
valid_idx <- which(lag_Resp >= 0 & stan_data$subj %in% selected_10_indist & lag_Ch > 0)

X_final <- X[valid_idx, ]
y_switch_final <- Switch[valid_idx]
subj_final <- stan_data$subj[valid_idx]
iti_final <- stan_data$ITI[valid_idx]
N_final <- length(valid_idx)

unique_subjs <- sort(unique(subj_final))
K_final <- length(unique_subjs)
subj_mapped <- numeric(N_final)
for (k in 1:K_final) { subj_mapped[subj_final == unique_subjs[k]] <- k }

ps <- readRDS("parameter_stability.rds")
calc_sd <- function(x_q3, x_q1) { max(abs(x_q3 - x_q1) / 1.35, 0.01) }

mu_alpha_pc = ps$Bio$alpha_pc[3];       sd_alpha_pc = calc_sd(ps$Bio$alpha_pc[3], ps$Bio$alpha_pc[1])
mu_lambda_pc = ps$Bio$lambda_pc[3];     sd_lambda_pc = calc_sd(ps$Bio$lambda_pc[3], ps$Bio$lambda_pc[1])
mu_beta_thal = ps$Bio$beta_thal[3];     sd_beta_thal = calc_sd(ps$Bio$beta_thal[3], ps$Bio$beta_thal[1])
mu_kappa_cf = ps$Bio$kappa_cf[3];       sd_kappa_cf = calc_sd(ps$Bio$kappa_cf[3], ps$Bio$kappa_cf[1])
mu_alpha_gran = ps$Bio$alpha_gran[3];   sd_alpha_gran = calc_sd(ps$Bio$alpha_gran[3], ps$Bio$alpha_gran[1])
mu_beta_gran = ps$Bio$beta_gran[3];     sd_beta_gran = calc_sd(ps$Bio$beta_gran[3], ps$Bio$beta_gran[1])

# ACHLIOPTAS MATRICES (2/3 sparsity)
generate_achlioptas <- function(rows, cols) {
  vals <- sample(c(sqrt(10), 0, -sqrt(10)), size = rows * cols, replace = TRUE, prob = c(1/20, 18/20, 1/20))
  return(matrix(vals / sqrt(cols), nrow = rows, ncol = cols))
}

set.seed(42)
W_gen <- generate_achlioptas(27, 32)
W_ach1 <- generate_achlioptas(896, 32)
W_ach2 <- generate_achlioptas(362, 896)
W_thal <- generate_achlioptas(32, 362)
Pi_vec <- rep(1, 27)

data_stan <- list(
  N = N_final,
  K = K_final,
  subj_id = subj_mapped,
  y = y_switch_final,
  X = X_final,
  iti = iti_final,
  W_gen = W_gen,
  W_ach1 = W_ach1,
  W_ach2 = W_ach2,
  W_thal = W_thal,
  Pi_vec = Pi_vec,
  prior_mu_alpha_pc = mu_alpha_pc,    prior_sd_alpha_pc = sd_alpha_pc,
  prior_mu_lambda_pc = mu_lambda_pc,  prior_sd_lambda_pc = sd_lambda_pc,
  prior_mu_beta_thal = mu_beta_thal,  prior_sd_beta_thal = sd_beta_thal,
  prior_mu_kappa_cf = mu_kappa_cf,    prior_sd_kappa_cf = sd_kappa_cf,
  prior_mu_alpha_gran = mu_alpha_gran, prior_sd_alpha_gran = sd_alpha_gran,
  prior_mu_beta_gran = mu_beta_gran,  prior_sd_beta_gran = sd_beta_gran
)

init_fun <- function() {
  list(
    alpha_pc_pop = mu_alpha_pc,
    lambda_pc_pop = mu_lambda_pc,
    beta_thal_pop = mu_beta_thal,
    kappa_cf_pop = mu_kappa_cf,
    alpha_gran_pop = mu_alpha_gran,
    beta_gran_pop = mu_beta_gran,
    sigma_alpha_pc = 0.01, sigma_lambda_pc = 0.01, sigma_beta_thal = 0.01,
    sigma_kappa_cf = 0.01, sigma_alpha_gran = 0.01, sigma_beta_gran = 0.01,
    alpha_pc_raw = rep(0, K_final), lambda_pc_raw = rep(0, K_final), beta_thal_raw = rep(0, K_final),
    kappa_cf_raw = rep(0, K_final), alpha_gran_raw = rep(0, K_final), beta_gran_raw = rep(0, K_final),
    alpha_pop = 0, sigma_alpha = 0.1, alpha_raw = rep(0, K_final),
    W_readout_pop = rep(0, 32), sigma_W_readout = rep(0.1, 32),
    W_readout_raw = matrix(0, nrow=32, ncol=K_final)
  )
}

cat("Compiling Full Hierarchical Bio Stan Model with SPARSE BPTT...\n")
mod_full <- cmdstan_model("full_bio_bptt.stan", 
                          stanc_options = list("allow-undefined"=TRUE),
                          cpp_options = list(USER_HEADER = file.path(getwd(), "bio_bptt.hpp")))

cat("Testing ONE EVALUATION speed using optimize for 90% SPARSE N=10...\n")
t0 <- Sys.time()
fit_opt <- mod_full$optimize(data = data_stan, init = init_fun, iter = 50, algorithm = "lbfgs")
t1 <- Sys.time()
print(t1 - t0)
