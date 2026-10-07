
library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(glmnet)

sourceCpp(code="
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
// [[Rcpp::export]]
List sim_true(int N_trials, arma::mat X, arma::vec ITI,
                 arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::mat Pi_mat,
                 double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                 double p_alpha_gran, double p_beta_gran, double p_sigma2_diff) {
    int active_gran = W_ach1.n_rows; int active_dcn = W_ach2.n_rows;
    arma::vec mu = arma::zeros(32); arma::vec Z = arma::zeros(active_gran); arma::vec W_purk = arma::zeros(active_gran); arma::vec D = arma::zeros(active_dcn); arma::vec G = arma::zeros(active_gran);
    arma::mat out_mu(N_trials, 32); arma::mat out_G(N_trials, active_gran); arma::mat out_Z(N_trials, active_gran); arma::mat out_W(N_trials, active_gran); arma::mat out_D(N_trials, active_dcn);
    for(int t = 0; t < N_trials; t++) {
        if(ITI[t] > 0) { Z.zeros(); W_purk += arma::randn<arma::vec>(active_gran) * std::sqrt(p_sigma2_diff * ITI[t]); }
        arma::vec I_t = X.row(t).t(); arma::vec I_hat = W_gen * mu; arma::vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        double decay = std::exp(-p_lambda_pc * ITI[t]); mu = mu * decay + p_alpha_pc * ((W_gen.t() * eps) + p_beta_thal * (W_thal * D)); mu = arma::clamp(mu, -50.0, 50.0);
        G = W_ach1 * mu; Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G; double eps_mag = arma::mean(arma::abs(eps)); W_purk = W_purk - p_kappa_cf * (eps_mag * Z); W_purk = arma::clamp(W_purk, -50.0, 50.0);
        arma::vec P = G % W_purk; D = W_ach2 * P; D = arma::clamp(D, -50.0, 50.0);
        out_mu.row(t) = mu.t(); out_G.row(t) = G.t(); out_Z.row(t) = Z.t(); out_W.row(t) = W_purk.t(); out_D.row(t) = D.t();
    }
    return List::create(Named(\"mu\")=out_mu, Named(\"G\")=out_G, Named(\"Z\")=out_Z, Named(\"W\")=out_W, Named(\"D\")=out_D);
}
")

targets <- readRDS("src/r/distillation_targets_v2.rds")
N_trials <- length(targets$labels); X <- targets$X[, 1:6]; ITI <- targets$ITI / max(targets$ITI); unique_subjs <- unique(targets$subjs)
Pi_mat <- matrix(0, nrow = N_trials, ncol = 6)
for (s in unique_subjs) {
  idx <- which(targets$subjs == s); subj_var <- apply(X[idx, ], 2, var); norm_prec <- (1.0 / (subj_var + 1e-6)) / mean(1.0 / (subj_var + 1e-6)); Pi_mat[idx, ] <- matrix(rep(norm_prec, length(idx)), nrow = length(idx), byrow = TRUE)
}
set.seed(42)
W_gen <- matrix(rnorm(6 * 32, 0, 1 / sqrt(32)), nrow = 6, ncol = 32); W_ach1_full <- matrix(rnorm(896 * 32, 0, 1 / sqrt(32)), nrow = 896, ncol = 32) * matrix(rbinom(896 * 32, 1, 0.1), nrow=896, ncol=32); W_ach2_full <- matrix(rnorm(896 * 896, 0, 1 / sqrt(896)), nrow = 896, ncol = 896); W_thal_full <- matrix(rnorm(32 * 896, 0, 1 / sqrt(896)), nrow = 32, ncol = 896)
N_GRAN <- 336; N_DCN <- 165; W_ach1 <- W_ach1_full[1:N_GRAN, ]; W_ach2 <- W_ach2_full[1:N_DCN, 1:N_GRAN]; W_thal <- W_thal_full[, 1:N_DCN]
b_p <- c(1.02043, 1.01043, 0.75440, 1.52653, 0.75645, 0.49589, 2.51338)

EMA_Left <- numeric(N_trials); Reward_Ent <- numeric(N_trials)
window_size <- 15; alpha_ema <- 2 / (5 + 1); current_subj <- -1
for (t in 1:N_trials) {
  if (targets$subjs[t] != current_subj) { current_ema_left <- 0.5; current_subj <- targets$subjs[t] }
  left_outcome <- ifelse(targets$lag_resp[t] == 1, targets$lag_reward[t], 1 - targets$lag_reward[t])
  current_ema_left <- alpha_ema * left_outcome + (1 - alpha_ema) * current_ema_left; EMA_Left[t] <- current_ema_left
}
for(s in unique_subjs) {
    idx <- which(targets$subjs == s); subj_rew <- targets$lag_reward[idx]
    for(t in 1:length(idx)) {
        if(t == 1) { p <- 0.5 } else if (t <= window_size) { p <- mean(subj_rew[1:t]) } else { p <- mean(subj_rew[(t - window_size + 1):t]) }
        p <- max(1e-6, min(1 - 1e-6, p)); Reward_Ent[idx[t]] <- -p*log2(p) - (1-p)*log2(1-p)
    }
}

Y_Act <- scale(targets$labels)
Y_Val <- scale(EMA_Left)
Y_Vol <- scale(Reward_Ent)

results <- data.frame()
cat("Running Synthetic Lesions and computing Cortical Readout Efficiency...\n")

for (s in unique_subjs) {
    idx <- which(targets$subjs == s)
    
    sim_intact <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, b_p[1], b_p[2], b_p[3], b_p[4], b_p[5], b_p[6], b_p[7])
    sim_thal <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, b_p[1], b_p[2], 0.0, b_p[4], b_p[5], b_p[6], b_p[7])
    sim_kappa <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, b_p[1], b_p[2], b_p[3], 0.0, b_p[5], b_p[6], b_p[7])
    
    conds <- list("Intact" = sim_intact$mu[idx,], 
                  "Lesion_Thal" = sim_thal$mu[idx,], 
                  "Lesion_Kappa" = sim_kappa$mu[idx,])
                  
    targs <- list("Action" = Y_Act[idx], "Value" = Y_Val[idx], "Volatility" = Y_Vol[idx])
    
    for(c_name in names(conds)) {
        mat <- scale(conds[[c_name]][, apply(conds[[c_name]], 2, sd) > 1e-8, drop=FALSE])
        
        pc1 <- prcomp(mat, center=TRUE, scale.=TRUE)$x[,1]
        volatility <- mean(abs(diff(pc1)))
        
        for(t_name in names(targs)) {
            Y_s <- targs[[t_name]]
            fit <- cv.glmnet(mat, Y_s, alpha=0, nfolds=5)
            preds <- predict(fit, newx=mat, s="lambda.min")[,1]
            r2 <- max(0, 1 - sum((Y_s - preds)^2) / sum((Y_s - mean(Y_s))^2))
            
            efficiency <- r2 / (volatility + 1e-6)
            
            results <- rbind(results, data.frame(
                Subject = s, Condition = c_name, Variable = t_name, 
                R2 = r2, Volatility = volatility, Efficiency = efficiency
            ))
        }
    }
}

summary_stats <- results %>% group_by(Variable, Condition) %>% summarise(
    Mean_Efficiency = mean(Efficiency),
    .groups="drop"
)

dir.create("results", showWarnings=FALSE)
sink("results/lesion_efficiency.txt")
cat("=== LESION ANALYSIS: CORTICAL READOUT EFFICIENCY ===\n")
print(as.data.frame(summary_stats), row.names=FALSE)
sink()
cat("Extraction done.\n")

