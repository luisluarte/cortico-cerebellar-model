
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(jsonlite)
})

cat("Loading N=100 Distilled Data & isolating 5 In-Distribution subjects...\n")
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
y_switch_final <- Switch[valid_idx]
subj_final <- stan_data$subj[valid_idx]
iti_final <- stan_data$ITI[valid_idx]
RT_final <- stan_data$RT[valid_idx]
N_final <- length(valid_idx)

subj_mapped <- numeric(N_final)
min_RT <- numeric(5)
for (k in 1:5) { 
  subj_mapped[subj_final == selected_5_indist[k]] <- k 
  min_RT[k] <- min(RT_final[subj_final == selected_5_indist[k]]) - 0.01 # slight buffer
}

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

ps <- readRDS("parameter_stability.rds")
if(is.data.frame(ps)) {
    q3_bio <- subset(ps, Model == "Bio" & Quartile == "Q3")
} else {
    q3_bio <- subset(ps$Bio, Quartile == "Q3")
}

data_joint <- list(
  N = N_final, K = 5, subj_id = subj_mapped, y = y_switch_final,
  X = X_final, iti = iti_final, RT = RT_final, min_RT = min_RT,
  W_gen = W_gen, W_ach1 = W_ach1, W_ach2 = W_ach2, W_thal = W_thal, Pi_vec = Pi_vec,
  prior_mu_alpha_pc = q3_bio$alpha_pc, prior_sd_alpha_pc = 0.1,
  prior_mu_lambda_pc = q3_bio$lambda_pc, prior_sd_lambda_pc = 0.1,
  prior_mu_beta_thal = q3_bio$beta_thal, prior_sd_beta_thal = 0.1,
  prior_mu_kappa_cf = q3_bio$kappa_cf, prior_sd_kappa_cf = 0.1,
  prior_mu_alpha_gran = q3_bio$alpha_gran, prior_sd_alpha_gran = 0.1,
  prior_mu_beta_gran = q3_bio$beta_gran, prior_sd_beta_gran = 0.1
)

cat("Compiling Joint Stan Model...\n")
mod_joint <- cmdstan_model("full_bio_joint.stan", 
                           stanc_options = list("allow-undefined"=TRUE),
                           cpp_options = list(USER_HEADER = file.path(getwd(), "bio_joint_bptt.hpp")))

init_joint <- function() {
  list(
    alpha_pc_pop_raw = -3, lambda_pc_pop_raw = -3, beta_thal_pop_raw = -3,
    kappa_cf_pop_raw = -3, alpha_gran_pop_raw = -3, beta_gran_pop_raw = 0,
    sigma_alpha_pc = 0.1, sigma_lambda_pc = 0.1, sigma_beta_thal = 0.1,
    sigma_kappa_cf = 0.1, sigma_alpha_gran = 0.1, sigma_beta_gran = 0.1,
    alpha_pc_raw = rep(0, 5), lambda_pc_raw = rep(0, 5), beta_thal_raw = rep(0, 5),
    kappa_cf_raw = rep(0, 5), alpha_gran_raw = rep(0, 5), beta_gran_raw = rep(0, 5),
    intercept_c_pop = 0, sigma_intercept_c = 0.1, intercept_c_raw = rep(0, 5),
    W_c_pop = rep(0, 32), sigma_W_c = rep(0.1, 32), W_c_raw = matrix(0, nrow=32, ncol=5),
    intercept_rt_pop = 0, sigma_intercept_rt = 0.1, intercept_rt_raw = rep(0, 5),
    W_rt_pop = rep(0, 32), sigma_W_rt = rep(0.01, 32), W_rt_raw = matrix(0, nrow=32, ncol=5),
    sigma_rt_pop_raw = 1, sigma_sigma_rt = 0.1, sigma_rt_raw = rep(0, 5),
    tau_rt_pop_raw = 0, sigma_tau_rt = 0.1, tau_rt_raw = rep(0, 5)
  )
}

cat("Fitting Joint Model via ADVI...\n")
fit_joint <- mod_joint$variational(data = data_joint, init = init_joint, seed = 123, iter = 2000, output_samples = 1000)
cat("SUCCESS! ADVI finished.\n")
# Save the fit
fit_joint$save_object("joint_sparse_fit.rds")
