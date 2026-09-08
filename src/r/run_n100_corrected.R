
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(loo)
  library(Rcpp)
  library(RcppArmadillo)
  library(jsonlite)
})

cat("Compiling C++ Simulators...\n")
sourceCpp("sim_bio.cpp")
sourceCpp(code = '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat simulate_qlearn(int N_trials, arma::vec subjs, arma::mat X, 
                          arma::vec lag_R, arma::vec lag_Resp,
                          double alpha_win, double alpha_loss) {
    arma::vec Q = arma::ones(8) * 0.5;
    arma::mat Q_history(N_trials, 8, arma::fill::zeros);
    double current_subj = -1;
    for(int t = 0; t < N_trials; t++) {
        if (subjs[t] != current_subj) {
            Q.fill(0.5);
            current_subj = subjs[t];
        }
        Q_history.row(t) = Q.t();
        
        int b1 = 0;
        for(int i = 0; i < 8; i++) { if(X(t, i) > 0.5) b1 = i; }
        int b2 = 0;
        for(int i = 8; i < 16; i++) { if(X(t, i) > 0.5) b2 = i - 8; }
        
        int chosen = (lag_Resp[t] == 1) ? b1 : b2;
        int unchosen = (lag_Resp[t] == 1) ? b2 : b1;
        
        if (lag_R[t] > 0) {
            Q[chosen] += alpha_win * (1.0 - Q[chosen]);
            Q[unchosen] += alpha_win * (0.0 - Q[unchosen]);
        } else {
            Q[chosen] += alpha_loss * (0.0 - Q[chosen]);
            Q[unchosen] += alpha_loss * (1.0 - Q[unchosen]);
        }
    }
    return Q_history;
}
')

cat("Loading N=100 Distilled Data...\n")
stan_data <- read_json("../../data/stan_data_N100_distilled.json", simplifyVector = TRUE)
N <- stan_data$N
N_subj <- stan_data$N_subj

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
    lag_Resp[i] <- 1
  }
}

X <- matrix(0, nrow = N, ncol = 27)
for (i in 1:N) {
  X[i, stan_data$Bd1[i]] <- 1
  X[i, stan_data$Bd2[i] + 8] <- 1
  X[i, 17] <- lag_Reward[i]
  X[i, 18] <- lag_RT[i]
  if (lag_Ch[i] > 0) {
    X[i, 18 + lag_Ch[i]] <- 1
  }
  X[i, 27] <- lag_Resp[i]
}

# IMPORTANT: We drop the first trial for each participant because it has no lag information!
valid_idx <- which(lag_Resp == 1)

cat(sprintf("Retained %d out of %d trials (dropped first trials).\n", length(valid_idx), N))

X_final <- X[valid_idx, ]
y_final <- Switch[valid_idx]
subj_final <- stan_data$subj[valid_idx]
iti_final <- stan_data$ITI[valid_idx]
lag_R_final <- lag_Reward[valid_idx]
lag_Resp_final <- lag_Resp[valid_idx]
N_final <- length(valid_idx)

ps <- readRDS("parameter_stability.rds")
if(is.data.frame(ps)) {
    q3_bio <- subset(ps, Model == "Bio" & Quartile == "Q3")
    q3_wsls <- subset(ps, Model == "WSLS" & Quartile == "Q3")
    q3_qlearn <- subset(ps, Model == "QLearn" & Quartile == "Q3")
} else {
    q3_bio <- subset(ps$Bio, Quartile == "Q3")
    q3_wsls <- subset(ps$WSLS, Quartile == "Q3")
    q3_qlearn <- subset(ps$QLearn, Quartile == "Q3")
}

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

cat("Simulating Traces...\n")
mu_hist <- simulate_bio(N_final, X_final, iti_final, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                        q3_bio$alpha_pc, q3_bio$lambda_pc, q3_bio$beta_thal, q3_bio$kappa_cf,
                        q3_bio$alpha_gran, q3_bio$beta_gran, q3_bio$sigma2_diff)

q_hist <- simulate_qlearn(N_final, subj_final, X_final, lag_R_final, lag_Resp_final,
                          q3_qlearn$alpha_win, q3_qlearn$alpha_loss)

cat("Loading Stan Models...\n")
mod_readout <- cmdstan_model("readout_pooled.stan")
mod_wsls <- cmdstan_model("wsls_pooled.stan")

cat("Running Stan Bio Model (Pooled, Corrected N=100)...\n")
data_bio <- list(N = N_final, D = 32, y = y_final, trace = mu_hist)
fit_bio <- mod_readout$sample(data = data_bio, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, refresh=500)
loo_bio <- fit_bio$loo()
cat("Bio ELPD:\n")
print(loo_bio)

cat("Running Stan QLearn Model (Pooled, Corrected N=100)...\n")
data_qlearn <- list(N = N_final, D = 8, y = y_final, trace = q_hist)
fit_qlearn <- mod_readout$sample(data = data_qlearn, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, refresh=500)
loo_qlearn <- fit_qlearn$loo()
cat("QLearn ELPD:\n")
print(loo_qlearn)

cat("Running Stan WSLS Model (Pooled, Corrected N=100)...\n")
data_wsls <- list(N = N_final, y = y_final,
                  win = ifelse(lag_R_final > 0, 1, 0),
                  q3_theta_win_logit = qlogis(q3_wsls$theta_win),
                  q3_theta_loss_logit = qlogis(q3_wsls$theta_loss))
fit_wsls <- mod_wsls$sample(data = data_wsls, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, refresh=500)
loo_wsls <- fit_wsls$loo()
cat("WSLS ELPD:\n")
print(loo_wsls)

comp <- loo_compare(list(Bio = loo_bio, WSLS = loo_wsls, QLearn = loo_qlearn))
cat("\n===========================\nLOO COMPARISON (Corrected N=100, Pooled)\n===========================\n")
print(comp)

saveRDS(list(Bio = loo_bio, WSLS = loo_wsls, QLearn = loo_qlearn, comp = comp), "final_elpd_n100_corrected.rds")
