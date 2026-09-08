
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))
suppressPackageStartupMessages({ library(cmdstanr); library(jsonlite) })

stan_data <- read_json("../../data/stan_data_N100_distilled.json", simplifyVector = TRUE)
train_subjs <- readRDS("rnn_phase1_subjects.rds")
N <- stan_data$N

Ch <- numeric(N)
for(i in 1:N) { Ch[i] <- ifelse(stan_data$Resp[i] == 1, stan_data$Bd1[i], stan_data$Bd2[i]) }

Switch <- numeric(N); lag_Reward <- numeric(N); lag_Ch <- numeric(N); lag_RT <- numeric(N); lag_Resp <- numeric(N)
current_subj <- -1
for(i in 1:N) {
  if (stan_data$subj[i] != current_subj) {
    Switch[i] <- 0; lag_Reward[i] <- 0; lag_Ch[i] <- 0; lag_RT[i] <- 0; lag_Resp[i] <- 0; current_subj <- stan_data$subj[i]
  } else {
    Switch[i] <- ifelse(Ch[i] != Ch[i-1], 1, 0)
    lag_Reward[i] <- stan_data$Reward[i-1]; lag_Ch[i] <- Ch[i-1]; lag_RT[i] <- stan_data$RT[i-1]; lag_Resp[i] <- ifelse(stan_data$Resp[i-1] == 1, 1, 0)
  }
}

X <- matrix(0, nrow = N, ncol = 27)
for (i in 1:N) {
  X[i, stan_data$Bd1[i]] <- 1; X[i, stan_data$Bd2[i] + 8] <- 1; X[i, 17] <- lag_Reward[i]; X[i, 18] <- lag_RT[i]
  if (lag_Ch[i] > 0) { X[i, 18 + lag_Ch[i]] <- 1 }; X[i, 27] <- lag_Resp[i]
}

set.seed(42)
selected_5_indist <- sample(train_subjs, 5)
valid_idx <- which(lag_Resp >= 0 & stan_data$subj %in% selected_5_indist & lag_Ch > 0)
X_final <- X[valid_idx, ]; y_switch_final <- Switch[valid_idx]; subj_final <- stan_data$subj[valid_idx]; iti_final <- stan_data$ITI[valid_idx]
N_final <- length(valid_idx)
unique_subjs <- sort(unique(subj_final)); K_final <- length(unique_subjs)
subj_mapped <- numeric(N_final)
for (k in 1:K_final) { subj_mapped[subj_final == unique_subjs[k]] <- k }

ps <- readRDS("parameter_stability.rds")
calc_sd <- function(x_q3, x_q1) { max(abs(x_q3 - x_q1) / 1.35, 0.01) }
mu_alpha_pc = ps$Bio$alpha_pc[3]; sd_alpha_pc = calc_sd(ps$Bio$alpha_pc[3], ps$Bio$alpha_pc[1])
mu_lambda_pc = ps$Bio$lambda_pc[3]; sd_lambda_pc = calc_sd(ps$Bio$lambda_pc[3], ps$Bio$lambda_pc[1])
mu_beta_thal = ps$Bio$beta_thal[3]; sd_beta_thal = calc_sd(ps$Bio$beta_thal[3], ps$Bio$beta_thal[1])
mu_kappa_cf = ps$Bio$kappa_cf[3]; sd_kappa_cf = calc_sd(ps$Bio$kappa_cf[3], ps$Bio$kappa_cf[1])
mu_alpha_gran = ps$Bio$alpha_gran[3]; sd_alpha_gran = calc_sd(ps$Bio$alpha_gran[3], ps$Bio$alpha_gran[1])
mu_beta_gran = ps$Bio$beta_gran[3]; sd_beta_gran = calc_sd(ps$Bio$beta_gran[3], ps$Bio$beta_gran[1])

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

data_stan <- list(
  N = N_final, K = K_final, subj_id = subj_mapped, y = y_switch_final, X = X_final, iti = iti_final,
  W_gen = W_gen, W_ach1 = W_ach1, W_ach2 = W_ach2, W_thal = W_thal, Pi_vec = Pi_vec,
  prior_mu_alpha_pc = mu_alpha_pc, prior_sd_alpha_pc = sd_alpha_pc,
  prior_mu_lambda_pc = mu_lambda_pc, prior_sd_lambda_pc = sd_lambda_pc,
  prior_mu_beta_thal = mu_beta_thal, prior_sd_beta_thal = sd_beta_thal,
  prior_mu_kappa_cf = mu_kappa_cf, prior_sd_kappa_cf = sd_kappa_cf,
  prior_mu_alpha_gran = mu_alpha_gran, prior_sd_alpha_gran = sd_alpha_gran,
  prior_mu_beta_gran = mu_beta_gran, prior_sd_beta_gran = sd_beta_gran
)

init_ncp <- function() {
  list(
    alpha_pc_pop = mu_alpha_pc, lambda_pc_pop = mu_lambda_pc, beta_thal_pop = mu_beta_thal,
    kappa_cf_pop = mu_kappa_cf, alpha_gran_pop = mu_alpha_gran, beta_gran_pop = mu_beta_gran,
    sigma_alpha_pc = 0.01, sigma_lambda_pc = 0.01, sigma_beta_thal = 0.01,
    sigma_kappa_cf = 0.01, sigma_alpha_gran = 0.01, sigma_beta_gran = 0.01,
    alpha_pc_raw = rep(0, K_final), lambda_pc_raw = rep(0, K_final), beta_thal_raw = rep(0, K_final),
    kappa_cf_raw = rep(0, K_final), alpha_gran_raw = rep(0, K_final), beta_gran_raw = rep(0, K_final),
    alpha_pop = 0, sigma_alpha = 0.1, alpha_raw = rep(0, K_final),
    W_readout_pop = rep(0, 32), sigma_W_readout = rep(0.1, 32), W_readout_raw = matrix(0, nrow=32, ncol=K_final)
  )
}

init_cp <- function() {
  list(
    alpha_pc_pop = mu_alpha_pc, lambda_pc_pop = mu_lambda_pc, beta_thal_pop = mu_beta_thal,
    kappa_cf_pop = mu_kappa_cf, alpha_gran_pop = mu_alpha_gran, beta_gran_pop = mu_beta_gran,
    sigma_alpha_pc = 0.01, sigma_lambda_pc = 0.01, sigma_beta_thal = 0.01,
    sigma_kappa_cf = 0.01, sigma_alpha_gran = 0.01, sigma_beta_gran = 0.01,
    alpha_pc_subj = rep(mu_alpha_pc, K_final), lambda_pc_subj = rep(mu_lambda_pc, K_final), 
    beta_thal_subj = rep(mu_beta_thal, K_final), kappa_cf_subj = rep(mu_kappa_cf, K_final), 
    alpha_gran_subj = rep(mu_alpha_gran, K_final), beta_gran_subj = rep(mu_beta_gran, K_final),
    alpha_pop = 0, sigma_alpha = 0.1, alpha_subj = rep(0, K_final),
    W_readout_pop = rep(0, 32), sigma_W_readout = rep(0.1, 32), W_readout_subj = matrix(0, nrow=32, ncol=K_final)
  )
}

mod_ncp <- cmdstan_model("full_bio_bptt.stan", stanc_options = list("allow-undefined"=TRUE), cpp_options = list(USER_HEADER = file.path(getwd(), "bio_bptt.hpp")))
mod_cp <- cmdstan_model("full_bio_bptt_cp.stan", stanc_options = list("allow-undefined"=TRUE), cpp_options = list(USER_HEADER = file.path(getwd(), "bio_bptt.hpp")))

results <- c("# Benchmark Results for Hierarchical BPTT Models (N=5)")

run_test <- function(name, code_block) {
  cat(sprintf("Running %s...\n", name))
  t0 <- Sys.time()
  res <- tryCatch({ eval(code_block); "Success" }, error = function(e) { paste("Failed:", e$message) })
  t1 <- Sys.time()
  diff <- as.numeric(difftime(t1, t0, units="secs"))
  iters_per_sec <- if(res == "Success") 20 / diff else NA
  line <- sprintf("- **%s**: %s | %.2f seconds (%.2f iters/sec)", name, res, diff, iters_per_sec)
  return(line)
}

r1 <- run_test("Model 1: Baseline HMC (NCP, Diagonal Metric)", quote({
  mod_ncp$sample(data = data_stan, init = init_ncp, chains = 1, iter_warmup = 20, iter_sampling = 0, max_treedepth = 7, refresh = 5)
}))
r2 <- run_test("Model 2: Baseline HMC (NCP, Dense Metric)", quote({
  mod_ncp$sample(data = data_stan, init = init_ncp, chains = 1, iter_warmup = 20, iter_sampling = 0, metric = "dense_e", max_treedepth = 7, refresh = 5)
}))
r3 <- run_test("Model 3: HMC Centered Parameterization (CP, Diagonal Metric)", quote({
  mod_cp$sample(data = data_stan, init = init_cp, chains = 1, iter_warmup = 20, iter_sampling = 0, max_treedepth = 7, refresh = 5)
}))
r4 <- run_test("Model 4: Laplace Approximation (Optimize + Hessian)", quote({
  fit_opt <- mod_ncp$optimize(data = data_stan, init = init_ncp, jacobian = FALSE, iter = 200)
  mod_ncp$laplace(data = data_stan, mode = fit_opt)
}))
r5 <- run_test("Model 5: Variational Inference (ADVI)", quote({
  mod_ncp$variational(data = data_stan, init = init_ncp, iter = 1000)
}))

results <- c(results, r1, r2, r3, r4, r5)
writeLines(results, "benchmark_results.md")
cat("Benchmark Complete!\n")
