
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

df_pers <- data.frame()
cat("Computing Perseveration Index...\n")
for (s in unique_subjs) {
    idx <- which(targets$subjs == s)
    sim_intact <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, b_p[1], b_p[2], b_p[3], b_p[4], b_p[5], b_p[6], b_p[7])
    sim_thal <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat, b_p[1], b_p[2], 0.0, b_p[4], b_p[5], b_p[6], b_p[7])
    
    Y_a <- targets$labels[idx]
    R_t <- targets$lag_reward[idx]
    
    ctx_in <- scale(sim_intact$mu[idx, apply(sim_intact$mu, 2, sd) > 1e-8, drop=FALSE])
    ctx_thal <- scale(sim_thal$mu[idx, apply(sim_thal$mu, 2, sd) > 1e-8, drop=FALSE])
    
    fit_in <- cv.glmnet(ctx_in, as.factor(Y_a), family="binomial", alpha=0, nfolds=5)
    pred_in <- predict(fit_in, newx=ctx_in, s="lambda.min", type="response")[,1]
    
    fit_thal <- cv.glmnet(ctx_thal, as.factor(Y_a), family="binomial", alpha=0, nfolds=5)
    pred_thal <- predict(fit_thal, newx=ctx_thal, s="lambda.min", type="response")[,1]
    
    A_in <- ifelse(pred_in > 0.5, 1, 0)
    A_thal <- ifelse(pred_thal > 0.5, 1, 0)
    
    # Calculate Lose-Stay (Perseveration)
    calc_ls <- function(A, R) {
        n <- length(A)
        stays <- (A[2:n] == A[1:(n-1)])
        losses <- (R[1:(n-1)] == 0)
        sum(stays & losses) / max(1, sum(losses))
    }
    
    # Calculate Autocorrelation of decision logits
    logit_in <- predict(fit_in, newx=ctx_in, s="lambda.min", type="link")[,1]
    logit_thal <- predict(fit_thal, newx=ctx_thal, s="lambda.min", type="link")[,1]
    
    df_pers <- rbind(df_pers, data.frame(
        Subject = s,
        Condition = c("Intact", "Lesion_Thal"),
        Lose_Stay = c(calc_ls(A_in, R_t), calc_ls(A_thal, R_t)),
        Autocorr = c(cor(logit_in[-1], logit_in[-length(logit_in)]), cor(logit_thal[-1], logit_thal[-length(logit_thal)]))
    ))
}

sink("results/perseveration_stats.txt")
cat("=== PERSEVERATION STATS ===\n")
library(lme4)
library(lmerTest)
library(emmeans)

m1 <- lmer(Lose_Stay ~ Condition + (1|Subject), data=df_pers)
cat("\n--- LOSE-STAY (Behavioral Perseveration) ---\n")
print(summary(m1)$coefficients)
print(emmeans(m1, pairwise ~ Condition)$contrasts)

m2 <- lmer(Autocorr ~ Condition + (1|Subject), data=df_pers)
cat("\n--- LOGIT AUTOCORRELATION (Latent Inertia) ---\n")
print(summary(m2)$coefficients)
print(emmeans(m2, pairwise ~ Condition)$contrasts)

sink()
cat("Done.\n")

