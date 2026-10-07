
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(glmnet)
library(patchwork)

sourceCpp(code="
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List sim_true(int N_trials, arma::mat X, arma::vec ITI,
                 arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::mat Pi_mat,
                 double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                 double p_alpha_gran, double p_beta_gran, double p_sigma2_diff) {
    
    int active_gran = W_ach1.n_rows;
    int active_dcn = W_ach2.n_rows;
    
    arma::vec mu = arma::zeros(32);
    arma::vec Z = arma::zeros(active_gran);
    arma::vec W_purk = arma::zeros(active_gran);
    arma::vec D = arma::zeros(active_dcn);
    arma::vec G = arma::zeros(active_gran);
    
    arma::mat out_mu(N_trials, 32);
    arma::mat out_G(N_trials, active_gran);
    arma::mat out_Z(N_trials, active_gran);
    arma::mat out_W(N_trials, active_gran);
    arma::mat out_D(N_trials, active_dcn);
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI[t] > 0) {
            Z.zeros();
            W_purk += arma::randn<arma::vec>(active_gran) * std::sqrt(p_sigma2_diff * ITI[t]);
        }
        arma::vec I_t = X.row(t).t();
        arma::vec I_hat = W_gen * mu;
        arma::vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        
        double decay = std::exp(-p_lambda_pc * ITI[t]);
        mu = mu * decay + p_alpha_pc * ((W_gen.t() * eps) + p_beta_thal * (W_thal * D));
        mu = arma::clamp(mu, -50.0, 50.0);
        
        G = W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = arma::mean(arma::abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        W_purk = arma::clamp(W_purk, -50.0, 50.0);
        
        arma::vec P = G % W_purk;
        D = W_ach2 * P;
        D = arma::clamp(D, -50.0, 50.0);
        
        out_mu.row(t) = mu.t();
        out_G.row(t) = G.t();
        out_Z.row(t) = Z.t();
        out_W.row(t) = W_purk.t();
        out_D.row(t) = D.t();
    }
    return List::create(Named(\"mu\")=out_mu, Named(\"G\")=out_G, Named(\"Z\")=out_Z, Named(\"W\")=out_W, Named(\"D\")=out_D);
}
")

cat("Loading exact NSGA-II environment...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
N_trials <- length(targets$labels)
X <- targets$X
ITI <- targets$ITI
ITI <- ITI / max(ITI)  # Normalize ITI as in NSGA-II

unique_subjs <- unique(targets$subjs)
# Generate trial index `nt`
df_meta <- data.frame(subj = targets$subjs)
df_meta <- df_meta %>% group_by(subj) %>% mutate(nt = row_number()) %>% ungroup()

cat("Constructing Inverse Variance Pi_mat...\n")
Pi_mat <- matrix(0, nrow = N_trials, ncol = 6)
for (s in unique_subjs) {
  idx <- which(targets$subjs == s)
  subj_X <- targets$X[idx, ]
  subj_var <- apply(subj_X, 2, var)
  raw_prec <- 1.0 / (subj_var + 1e-6)
  norm_prec <- raw_prec / mean(raw_prec)
  Pi_mat[idx, ] <- matrix(rep(norm_prec, length(idx)), nrow = length(idx), byrow = TRUE)
}

set.seed(42)
W_gen <- matrix(rnorm(6 * 32, 0, 1 / sqrt(32)), nrow = 6, ncol = 32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1 / sqrt(32)), nrow = 896, ncol = 32) * matrix(rbinom(896 * 32, 1, 0.1), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(362 * 896, 0, 1 / sqrt(896)), nrow = 362, ncol = 896)
W_thal <- matrix(rnorm(32 * 362, 0, 1 / sqrt(362)), nrow = 32, ncol = 362)

# NSGA-II Params
b_p <- c(1.02043, 1.01043, 0.75440, 1.52653, 0.75645, 0.49589, 2.51338)

cat("Running C++ bio-simulation...\n")
sim <- sim_true(N_trials, X, ITI, W_gen, W_ach1, W_ach2, W_thal, Pi_mat,
                b_p[1], b_p[2], b_p[3], b_p[4], b_p[5], b_p[6], b_p[7])

cat("Computing trial-by-trial Q-learning targets...\n")
State_t <- numeric(N_trials); Value_t <- numeric(N_trials); Uncert_t <- numeric(N_trials)
Q1 <- 0.5; Q2 <- 0.5; alpha <- 0.2; beta <- 3.0

current_subj <- -1
stan_data_subj <- targets$subjs
stan_data_bd1 <- targets$X[, 1] # Magnitude Left
stan_data_bd2 <- targets$X[, 2] # Magnitude Right
stan_data_resp <- targets$lag_resp
stan_data_reward <- targets$lag_reward

for (t in 1:N_trials) {
  if (stan_data_subj[t] != current_subj) { Q1 <- 0.5; Q2 <- 0.5; current_subj <- stan_data_subj[t] }
  
  ev_left <- Q1 * stan_data_bd1[t]
  ev_right <- Q2 * stan_data_bd2[t]
  p_bd1 <- plogis(beta * (ev_left - ev_right))
  
  State_t[t] <- p_bd1
  Value_t[t] <- p_bd1 * Q1 + (1 - p_bd1) * Q2
  ent <- 0; if(p_bd1>0) ent <- ent - p_bd1*log2(p_bd1); if((1-p_bd1)>0) ent <- ent - (1-p_bd1)*log2(1-p_bd1)
  Uncert_t[t] <- ent
  
  chosen <- ifelse(stan_data_resp[t] == 1, 1, 2)
  unchosen <- ifelse(stan_data_resp[t] == 1, 2, 1)
  r <- stan_data_reward[t]
  if (r > 0) { Q1 <- ifelse(chosen==1, Q1 + alpha*(1-Q1), Q1); Q2 <- ifelse(chosen==2, Q2 + alpha*(1-Q2), Q2) } 
  else { Q1 <- ifelse(chosen==1, Q1 + alpha*(0-Q1), Q1); Q2 <- ifelse(chosen==2, Q2 + alpha*(0-Q2), Q2) }
}

df_meta$State <- State_t; df_meta$Value <- Value_t; df_meta$Uncert <- Uncert_t

layers <- list("1. Cortical (mu)" = sim$mu, 
               "2. Granule (G)" = sim$G,
               "3. Granule Trace (Z)" = sim$Z,
               "4. Purkinje (W)" = sim$W, 
               "5. DCN (D)" = sim$D)

targs <- c("State", "Value", "Uncertainty")
window_size <- 5; step_size <- 1; max_nt <- max(df_meta$nt, na.rm=TRUE)

results <- data.frame()
cat("Running 5-Trial Rolling Ridge with TRUE Environment...\n")

for(t_end in seq(window_size, max_nt, by = step_size)) {
    idx <- which(df_meta$nt > (t_end - window_size) & df_meta$nt <= t_end)
    if(length(idx) < 30) next 
    
    for(l_name in names(layers)) {
        X_full <- layers[[l_name]]
        X_win <- X_full[idx, , drop=FALSE]
        sd_cols <- apply(X_win, 2, sd)
        X_win <- X_win[, sd_cols > 1e-8, drop=FALSE]
        if(ncol(X_win) < 2) next
        
        for(t_name in targs) {
            col_name <- if(t_name == "Uncertainty") "Uncert" else t_name
            Y_win <- df_meta[[col_name]][idx]
            
            fit <- tryCatch({cv.glmnet(X_win, Y_win, alpha=0, standardize=TRUE, nfolds=5)}, error = function(e) NULL)
            
            if(!is.null(fit)) {
                r2 <- max(0, 1 - (min(fit$cvm) / var(Y_win)))
                results <- rbind(results, data.frame(Trial_End=t_end, Layer=l_name, Encoding=t_name, R2=r2))
            }
        }
    }
}

p <- ggplot(results, aes(x=Trial_End, y=R2, color=Encoding)) +
    geom_line(linewidth=1.2, alpha=0.8) +
    facet_wrap(~Layer, ncol=2, scales="free_y") +
    scale_color_manual(values=c("State"="dodgerblue", "Value"="firebrick", "Uncertainty"="forestgreen")) +
    labs(title="TRUE High-Resolution Rolling Ridge (5-Trial Window)", 
         subtitle="Decoding dynamics using the exact NSGA-II precision-weighted environment",
         y="Cross-Validated R-squared", x="Trial Number (Window End)") +
    theme_minimal(base_size=14) + theme(legend.position="top")

ggsave("results/Fig62_True_Rolling_Ridge_5.png", plot=p, width=14, height=10, dpi=300)
cat("Done.\n")

