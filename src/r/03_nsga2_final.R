
library(Rcpp)
library(RcppArmadillo)
library(mco)

cat("Starting 03_nsga2_final.R...\n")
sourceCpp("nsga2_obj.cpp")

d <- readRDS("../../data/dataset_A.rds")
targets <- readRDS("../../results/oracle_targets.rds")

X <- as.matrix(d[, c("Bd1_scaled", "Bd2_scaled", "lag_Reward", "lag_Ch", "lag_Resp", "ITI_scaled")])
if(ncol(X) < 6) X <- cbind(X, matrix(0, nrow=nrow(X), ncol=6-ncol(X)))
ITI <- d$ITI_scaled
reward <- d$lag_Reward
choices <- d$Resp_mapped
y_rt <- d$RT_scaled

subjs <- unique(d$participant_id)
rnn_mu <- matrix(0, nrow=nrow(d), ncol=ncol(targets[[1]]$hidden))
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

fitness_fn <- function(params_with_gamma) {
  params <- params_with_gamma[1:7]
  gamma <- params_with_gamma[8]
  
  L_bounds <- c(0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000)
  U_bounds <- c(1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000)
  params_scaled <- L_bounds + (U_bounds - L_bounds) * plogis(params)
  
  res <- eval_nsga2_obj(params_scaled, length(subjs), subj_indices, X, ITI, reward, choices, y_rt, rnn_mu, gamma, W_gen, W_ach1, W_ach2, W_thal, Pi_mat)
  if(is.nan(res[1]) || res[1] > 1e9) return(c(1e9, 0))
  return(res)
}

cat("Running Full Rigorous NSGA-II...\n")
# Rigorous extraction
res <- nsga2(fitness_fn, idim=8, odim=2, 
             lower.bounds=c(rep(-5, 7), 0), 
             upper.bounds=c(rep(5, 7), 1),
             popsize=100, generations=100) 

cat("NSGA-II Complete.\n")
pareto_front <- res$value
pareto_params <- res$par

valid_idx <- which(pareto_front[,1] < 1e8)
if(length(valid_idx) == 0) valid_idx <- 1:nrow(pareto_front)
best_idx <- valid_idx[which.min(abs(pareto_params[valid_idx, 8] - 0.5))]

mu_EB <- pareto_params[best_idx, 1:7]
saveRDS(mu_EB, "../../results/mu_EB_final.rds")
cat("Saved mu_EB_final.rds!\n")
