# setup ----

# set local lib path
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# set download preferences
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://cloud.r-project.org"
  options(repos = r)
})

# install package manager
cat("\n############ installing pacman #############\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman", lib = local_lib)
}

cat("\n###### loading libs #########\n")
pacman::p_load(
  tidyverse,
  Rcpp,
  RcppArmadillo,
  parallel,
  doParallel,
  foreach
)

cli::cli_h2("setting workspace dir")
if (dir.exists("src/r")) {
  setwd("src/r")
}
cli::cli_text(getwd())

cli::cli_alert_info("compiling biological simulator...")
sourceCpp("sim_bio_probes.cpp")

cli::cli_alert_info("loading distillation targets...")
targets <- readRDS("distillation_targets_v2.rds")
N_trials <- length(targets$labels)
ITI <- targets$ITI
rnn_logits <- targets$rnn_logits
p_rnn <- plogis(rnn_logits)
p_emp <- targets$labels
subjs <- targets$subjs

# CRITICAL FIX: The true biological model was optimized for a 6-dimensional X!
# We must extract the 6 continuous scaled dimensions, or the weights will explode.
Bd1_val <- apply(targets$X[, 1:8], 1, function(r) { w <- which(r==1); if(length(w)>0) w[1] else 1 })
Bd1_scaled <- (Bd1_val - min(Bd1_val)) / (max(Bd1_val) - min(Bd1_val))

Bd2_val <- apply(targets$X[, 9:16], 1, function(r) { w <- which(r==1); if(length(w)>0) w[1] else 1 })
Bd2_scaled <- (Bd2_val - min(Bd2_val)) / (max(Bd2_val) - min(Bd2_val))

ITI_scaled <- targets$ITI / max(targets$ITI)

X <- as.matrix(cbind(Bd1_scaled, Bd2_scaled, targets$lag_reward, targets$lag_resp, targets$lag_resp, ITI_scaled))
input_dim <- 6

cli::cli_alert_info("loading baseline parameters (to fix non-grid axes to their medians)...")
post_df <- readRDS("../../results/factorial_fixed_10k_A.rds")
all_subj_params <- t(sapply(post_df[["bio_dist"]][["traces"]], function(x) exp(apply(x, 3, median))))
fixed_p <- apply(all_subj_params, 2, median)

# fixed parameters (indexes 2 through 7)
p_lambda_pc <- fixed_p[2]
p_beta_thal <- fixed_p[3]
p_alpha_gran <- fixed_p[5]
p_beta_gran <- fixed_p[6]
p_sigma2_diff <- fixed_p[7]

# for the projection layer (W_gen, W_ach, etc.)
# CRITICAL FIX: The matrices MUST be exactly 896 (Granule) and 362 (DCN) to match the MCMC optimized parameters!
set.seed(42)
W_gen <- matrix(rnorm(input_dim * 32, 0, 1 / sqrt(32)), nrow = input_dim, ncol = 32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1 / sqrt(32)), nrow = 896, ncol = 32)
mask1 <- matrix(rbinom(896 * 32, 1, 0.1), nrow = 896, ncol = 32)
W_ach1 <- W_ach1 * mask1
W_ach2 <- matrix(rnorm(362 * 896, 0, 1 / sqrt(896)), nrow = 362, ncol = 896)
W_thal <- matrix(rnorm(32 * 362, 0, 1 / sqrt(362)), nrow = 32, ncol = 362)
Pi_mat <- matrix(1, nrow = N_trials, ncol = input_dim)

cli::cli_alert_info("Defining Landscape Grid...")
alpha_pc_seq <- exp(seq(log(0.0005), log(1.5), length.out = 50))
kappa_cf_seq <- exp(seq(log(0.0001), log(0.1), length.out = 50))
grid <- expand.grid(alpha_pc = alpha_pc_seq, kappa_cf = kappa_cf_seq)

cli::cli_alert_info("starting parallel evaluation over {nrow(grid)} grid points...")
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

work_dir <- getwd()
parallel::clusterExport(cl, varlist = "work_dir", envir = environment())
parallel::clusterEvalQ(cl, {
  setwd(work_dir)
  library(Rcpp)
  library(RcppArmadillo)
  sourceCpp("sim_bio_probes.cpp")
})

# evaluate each grid point
results_list <- foreach(
  i = 1:nrow(grid),
  .combine = rbind,
  .noexport = c("simulate_bio_probes"),
  .export = c(
    "grid", "N_trials", "X", "ITI", "W_gen", "W_ach1", "W_ach2", "W_thal",
    "Pi_mat", "p_lambda_pc", "p_beta_thal", "p_alpha_gran", "p_beta_gran", "p_sigma2_diff",
    "p_emp", "p_rnn", "rnn_logits"
  ),
  .packages = c("Rcpp", "RcppArmadillo")
) %dopar% {
  alpha_val <- grid$alpha_pc[i]
  kappa_val <- grid$kappa_cf[i]

  # run the full ODE integration
  mu_history <- simulate_bio_probes(
    N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat,
    alpha_val, p_lambda_pc, p_beta_thal, kappa_val,
    p_alpha_gran, p_beta_gran, p_sigma2_diff, 0.1, 0.1
  )$mu

  # if simulation explodes or contains NA, return worst-case loss
  if (any(is.na(mu_history)) || any(is.nan(mu_history)) || any(is.infinite(mu_history))) {
    return(data.frame(
      alpha_pc = alpha_val, kappa_cf = kappa_val,
      empirical_nll = 100.0, rnn_nll = 100.0
    ))
  }

  # fit the linear readout dynamically (Optimal Linear Readout)
  XX_inv <- tryCatch(solve(crossprod(mu_history) + diag(0.01, ncol(mu_history))), error = function(e) NULL)

  if (is.null(XX_inv)) {
    return(data.frame(
      alpha_pc = alpha_val, kappa_cf = kappa_val,
      empirical_nll = 100.0, rnn_nll = 100.0
    ))
  }

  # CRITICAL FIX: The readout weights MUST be optimized against the Teacher RNN!
  # If you optimize them against the human labels, the landscape becomes degenerate and jagged.
  W_policy <- XX_inv %*% crossprod(mu_history, rnn_logits)
  bio_logits <- as.numeric(mu_history %*% W_policy)

  bio_probs <- plogis(bio_logits)
  bio_probs <- pmax(pmin(bio_probs, 1 - 1e-7), 1e-7)

  # empirical nll
  emp_nll <- -mean(p_emp * log(bio_probs) + (1 - p_emp) * log(1 - bio_probs))

  # rnn nll
  rnn_nll_val <- -mean(p_rnn * log(bio_probs) + (1 - p_rnn) * log(1 - bio_probs))

  data.frame(
    alpha_pc = alpha_val, kappa_cf = kappa_val,
    empirical_nll = emp_nll, rnn_nll = rnn_nll_val
  )
}

stopCluster(cl)

out_dir <- if (dir.exists("../../results")) "../../results" else if (dir.exists("results")) "results" else "."
out_file <- file.path(out_dir, "landscape_data_remote.rds")
saveRDS(results_list, out_file)
cli::cli_alert_success("grid computation complete! saved to {out_file}")
