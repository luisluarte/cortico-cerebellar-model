
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(loo)
  library(Rcpp)
  library(RcppArmadillo)
})

cat("Compiling C++ Simulators...\n")
sourceCpp("sim_bio.cpp")

cat("Loading Data...\n")
targets <- readRDS("distillation_targets_v2.rds")
ps <- readRDS("parameter_stability.rds")

if(is.data.frame(ps)) {
    q3_bio <- subset(ps, Model == "Bio" & Quartile == "Q3")
    q3_wsls <- subset(ps, Model == "WSLS" & Quartile == "Q3")
} else {
    q3_bio <- subset(ps$Bio, Quartile == "Q3")
    q3_wsls <- subset(ps$WSLS, Quartile == "Q3")
}

unique_subjs <- unique(targets$subjs)
pilot_subjs <- unique_subjs[1:5]
idx <- which(targets$subjs %in% pilot_subjs)

subjs <- targets$subjs[idx]
subj_id <- as.numeric(as.factor(subjs))
N <- length(idx)
K <- 5
y <- targets$labels[idx]

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

cat("Simulating Bio Trace...\n")
mu_hist <- simulate_bio(N, targets$X[idx, ], targets$ITI[idx], W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                        q3_bio$alpha_pc, q3_bio$lambda_pc, q3_bio$beta_thal, q3_bio$kappa_cf,
                        q3_bio$alpha_gran, q3_bio$beta_gran, q3_bio$sigma2_diff)

cat("Compiling Stan Models...\n")
mod_readout <- cmdstan_model("readout.stan")
mod_wsls <- cmdstan_model("wsls.stan")

cat("Running Stan Bio Model...\n")
data_bio <- list(N = N, K = K, D = 32, subj_id = subj_id, y = y, trace = mu_hist)
fit_bio <- mod_readout$sample(data = data_bio, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000)

loo_bio <- fit_bio$loo()
cat("Bio ELPD:\n")
print(loo_bio)

cat("Running Stan WSLS Model...\n")
data_wsls <- list(N = N, K = K, subj_id = subj_id, y = y,
                  win = ifelse(targets$lag_reward[idx] > 0, 1, 0),
                  lag_c = targets$lag_ch[idx],
                  q3_theta_win_logit = qlogis(q3_wsls$theta_win),
                  q3_theta_loss_logit = qlogis(q3_wsls$theta_loss))
fit_wsls <- mod_wsls$sample(data = data_wsls, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000)

loo_wsls <- fit_wsls$loo()
cat("WSLS ELPD:\n")
print(loo_wsls)

comp <- loo_compare(loo_bio, loo_wsls)
cat("\n===========================\nLOO COMPARISON\n===========================\n")
print(comp)
saveRDS(comp, "pilot_loo_compare.rds")
