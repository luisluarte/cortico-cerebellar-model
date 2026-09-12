
library(Rcpp)
library(RcppArmadillo)
library(parallel)

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")
mu_EB <- readRDS("../../results/mu_EB_static.rds")
X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled", "lag_Reward", "lag_Ch", "lag_Resp", "ITI_scaled")])
if(ncol(X) < 6) X <- cbind(X, matrix(0, nrow=nrow(X), ncol=6-ncol(X)))
ITI <- d$ITI_scaled
reward <- d$lag_Reward
choices <- d$Resp_mapped
y_rt <- d$RT_scaled
subjs <- unique(d$participant_id)
target_hidden_dim <- ncol(targets[[1]]$hidden)
rnn_mu <- matrix(0, nrow=nrow(d), ncol=target_hidden_dim)
subj_indices <- numeric(nrow(d))
idx_counter <- 1
for(i in 1:length(subjs)) {
  s <- subjs[i]
  n <- sum(d$participant_id == s)
  rng <- idx_counter:(idx_counter + n - 1)
  rnn_mu[rng, ] <- targets[[s]]$hidden
  subj_indices[rng] <- i - 1
  idx_counter <- idx_counter + n
}
set.seed(42)
W_gen <- matrix(rnorm(6 * 32, 0, 1/sqrt(32)), nrow=6, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1/sqrt(896)), nrow=362, ncol=896)
W_thal <- matrix(rnorm(32 * 362, 0, 1/sqrt(362)), nrow=32, ncol=362)
Pi_mat <- matrix(0, nrow=nrow(d), ncol=6)
for(s in 0:(length(subjs)-1)) {
  s_idx <- which(subj_indices == s)
  subj_var <- apply(X[s_idx, ], 2, var)
  raw_prec <- 1.0 / (subj_var + 1e-6)
  Pi_mat[s_idx, ] <- matrix(rep(raw_prec / mean(raw_prec), length(s_idx)), nrow=length(s_idx), byrow=TRUE)
}
sourceCpp("bio_mcmc_chain.cpp")

tasks <- expand.grid(subj_id = 0:1, chain = 1:2)
res <- mclapply(1:nrow(tasks), function(i) {
    s <- tasks$subj_id[i]
    s_idx <- which(subj_indices == s)
    out <- tryCatch({
      run_bio_mcmc_chain(10, X[s_idx,], ITI[s_idx], reward[s_idx], choices[s_idx], y_rt[s_idx], rnn_mu[s_idx,], "empirical", "flat", mu_EB, W_gen, W_ach1, W_ach2, W_thal, Pi_mat[s_idx,])
    }, error=function(e) as.character(e))
    return(out)
}, mc.cores=4)

for(i in 1:length(res)) {
  if(is.character(res[[i]])) {
    cat(sprintf("Task %d Error: %s\n", i, res[[i]]))
  }
}
cat("Done testing mclapply error\n")
