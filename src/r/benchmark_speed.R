local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(mco)
})

targets <- readRDS("distillation_targets_v2.rds")

N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
rnn_p_logits <- targets$rnn_logits
rnn_mu <- targets$rnn_mu
rnn_sigma <- targets$rnn_sigma
p_val <- plogis(rnn_p_logits) # Dummy for speed test

set.seed(42)
W_gen <- matrix(rnorm(27 * 32, 0, 1/sqrt(32)), nrow=27, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_vec <- rep(1, 27)

# Dummy parameters
p_alpha_pc <- 0.5
p_lambda_pc <- 0.05
p_beta_thal <- 0.01
p_kappa_cf <- 0.5
p_alpha_gran <- 0.5
p_beta_gran <- 0.5
p_sigma2_diff <- 0.01
gamma <- 0.5

start_time <- Sys.time()

mu <- rep(0, 32); Z <- rep(0, 896); W_purk <- rep(0, 896); D <- rep(0, 362)
mu_history <- matrix(0, nrow=N_trials, ncol=32)
diverged <- FALSE

for (t in 1:N_trials) {
  if (ITI[t] > 0) {
    Z <- rep(0, 896)
    W_purk <- W_purk + rnorm(896, 0, sqrt(p_sigma2_diff * ITI[t]))
  }
  I_t <- X[t, ]
  I_hat <- as.numeric(W_gen %*% mu)
  eps <- Pi_vec * (I_t - I_hat)
  mu <- mu + p_alpha_pc * (as.numeric(t(W_gen) %*% eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * as.numeric(W_thal %*% D))
  
  G <- as.numeric(W_ach1 %*% mu)
  Z <- (1.0 - p_beta_gran) * Z + p_alpha_gran * G
  eps_mag <- mean(abs(eps))
  W_purk <- W_purk - p_kappa_cf * (eps_mag * Z)
  D <- as.numeric(W_ach2 %*% (G * W_purk))
  mu_history[t, ] <- mu
}

XX_inv <- solve(crossprod(mu_history) + diag(0.01, 32))
W_policy <- XX_inv %*% crossprod(mu_history, rnn_p_logits)
W_mu <- XX_inv %*% crossprod(mu_history, rnn_mu)

bio_logits <- as.numeric(mu_history %*% W_policy)
bio_mu <- mu_history %*% W_mu

blend_logits <- (1 - gamma) * rnn_p_logits + gamma * bio_logits
blend_mu <- (1 - gamma) * rnn_mu + gamma * bio_mu

kinematic_loss <- mean( (rnn_mu - blend_mu)^2 / (2 * rnn_sigma^2) )
blend_probs <- plogis(blend_logits)
bce <- -mean(p_val * log(blend_probs) + (1 - p_val) * log(1 - blend_probs))

end_time <- Sys.time()
cat("Time taken:", as.numeric(difftime(end_time, start_time, units="secs")), "seconds\n")
