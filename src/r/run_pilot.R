
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(rstan)
  library(loo)
  library(Rcpp)
  library(RcppArmadillo)
})

# Optimize Stan for parallel execution
options(mc.cores = 4)
rstan_options(auto_write = TRUE)

cat("Compiling C++ Simulators...\n")
sourceCpp("sim_bio.cpp")

# Basic Q-Learning Simulator
sourceCpp(code = '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
// [[Rcpp::export]]
arma::mat simulate_qlearn(int N_trials, arma::vec lag_R, arma::vec lag_C, arma::vec lag_Resp,
                          double p_alpha_win, double p_alpha_loss, double p_beta) {
    arma::vec Q = arma::zeros(8);
    arma::mat Q_history(N_trials, 8, arma::fill::zeros);
    for(int t = 0; t < N_trials; t++) {
        if(lag_Resp[t] == 1) {
            double alpha = (lag_R[t] > 0) ? p_alpha_win : p_alpha_loss;
            int state = -1;
            int b1 = 1; int b2 = 1;
            if(lag_C[t] == b1) state = 0;
            else if(lag_C[t] == b2) state = 1;
            else state = 2; // simplified
            
            // Just generic mapping to 8 states, we will use simplified for benchmark if needed
            // Actually, wait, QLearn in original target script was mapped correctly.
            // For Pilot, we can just use the target matrix
        }
        Q_history.row(t) = Q.t();
    }
    return Q_history;
}
')

cat("Loading Data...\n")
targets <- readRDS("distillation_targets_v2.rds")
ps <- readRDS("parameter_stability.rds")

# Extract Q3 parameters
q3_bio <- subset(ps, Model == "Bio" & Quartile == "Q3")
q3_qlearn <- subset(ps, Model == "QLearn" & Quartile == "Q3")
q3_wsls <- subset(ps, Model == "WSLS" & Quartile == "Q3")

# Pilot Subset N=5
unique_subjs <- unique(targets$subjs)
pilot_subjs <- unique_subjs[1:5]
idx <- which(targets$subjs %in% pilot_subjs)

# Prepare Data
subjs <- targets$subjs[idx]
subj_id <- as.numeric(as.factor(subjs))
N <- length(idx)
K <- 5
y <- targets$labels[idx]

# Biological initialization
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

cat("Running Stan Bio Model...\n")
data_bio <- list(N = N, K = K, D = 32, subj_id = subj_id, y = y, trace = mu_hist)
fit_bio <- stan(file = "readout.stan", data = data_bio, chains = 4, iter = 2000, warmup = 1000, refresh=500)
loo_bio <- loo(fit_bio)
cat("Bio ELPD:\n")
print(loo_bio)

cat("Running Stan WSLS Model...\n")
data_wsls <- list(N = N, K = K, subj_id = subj_id, y = y,
                  win = ifelse(targets$lag_reward[idx] > 0, 1, 0),
                  lag_c = targets$lag_ch[idx],
                  q3_theta_win_logit = qlogis(q3_wsls$theta_win),
                  q3_theta_loss_logit = qlogis(q3_wsls$theta_loss))
fit_wsls <- stan(file = "wsls.stan", data = data_wsls, chains = 4, iter = 2000, warmup = 1000, refresh=500)
loo_wsls <- loo(fit_wsls)
cat("WSLS ELPD:\n")
print(loo_wsls)

comp <- loo_compare(loo_bio, loo_wsls)
cat("\n===========================
LOO COMPARISON\n===========================\n")
print(comp)
saveRDS(comp, "pilot_loo_compare.rds")
