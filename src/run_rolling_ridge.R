
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(dplyr)
library(glmnet)

sourceCpp(code="
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List sim_5layers_patched(int N_trials, arma::mat X, arma::vec ITI,
                 arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::mat Pi_mat,
                 double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                 double p_alpha_gran, double p_beta_gran, double p_sigma2_diff,
                 double p_gran_ratio, double p_dcn_ratio) {
    
    int max_gran = W_ach1.n_rows;
    int max_dcn = W_ach2.n_rows;
    int active_gran = std::round(p_gran_ratio * max_gran);
    int active_dcn = std::round(p_dcn_ratio * max_dcn);
    if(active_gran < 2) active_gran = 2;
    if(active_dcn < 2) active_dcn = 2;
    
    arma::vec mu = arma::zeros(32);
    arma::vec Z = arma::zeros(active_gran);
    arma::vec W_purk = arma::zeros(active_gran);
    arma::vec D = arma::zeros(active_dcn);
    arma::vec G = arma::zeros(active_gran);
    
    arma::mat sub_W_ach1 = W_ach1.rows(0, active_gran - 1);
    arma::mat sub_W_ach2 = W_ach2.submat(0, 0, active_dcn - 1, active_gran - 1);
    arma::mat sub_W_thal = W_thal.cols(0, active_dcn - 1);
    
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
        mu = mu * decay + p_alpha_pc * ((W_gen.t() * eps) + p_beta_thal * (sub_W_thal * D));
        mu = arma::clamp(mu, -50.0, 50.0);
        
        G = sub_W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = arma::mean(arma::abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        W_purk = arma::clamp(W_purk, -50.0, 50.0);
        
        arma::vec P = G % W_purk;
        D = sub_W_ach2 * P;
        D = arma::clamp(D, -50.0, 50.0);
        
        out_W.row(t) = W_purk.t();
        out_D.row(t) = D.t();
    }
    return List::create(Named(\"W\")=out_W, Named(\"D\")=out_D);
}
")

b_p <- c(1.02043, 1.01043, 0.75440, 1.52653, 0.75645, 0.49589, 2.51338, 0.375, 0.184)
set.seed(42)
W_gen <- matrix(rnorm(128 * 32, 0, 1/sqrt(32)), nrow=128, ncol=32)
W_ach1 <- matrix(rnorm(896 * 32, 0, 1/sqrt(32)), nrow=896, ncol=32) * matrix(rbinom(896 * 32, 1, 0.1), nrow=896, ncol=32)
W_ach2 <- matrix(rnorm(896 * 896, 0, 1/sqrt(896)), nrow=896, ncol=896)
W_thal <- matrix(rnorm(32 * 896, 0, 1/sqrt(896)), nrow=32, ncol=896)
v_A <- rep(0, 128); v_A[sample(1:128, 32)] <- 1.0
v_B <- rep(0, 128); v_B[sample(1:128, 32)] <- 1.0

d <- read.csv("data/behavioral_compilate.csv") %>% filter(Resp %in% c(1, 2))
p_ids <- unique(d$participant_id)[1:20] 

res_RL <- list(); res_W <- list(); res_D <- list()

cat("Simulating Exact Empirical Sequences...\n")
for(s_idx in seq_along(p_ids)) {
    pid <- p_ids[s_idx]
    p_data <- d %>% filter(participant_id == pid) %>% arrange(nt)
    
    ITI_sec <- pmax(pmin((p_data$ttp - lag(p_data$ttF, default = p_data$ttp[1] - 3000)) / 1000, 20), 0)
    probs <- p_data$prob
    n_trials <- nrow(p_data)
    
    Q1 <- 0.5; Q2 <- 0.5; alpha <- 0.2; beta <- 3.0
    State_t <- numeric(n_trials); Value_t <- numeric(n_trials); Uncert_t <- numeric(n_trials)
    seq_idx <- numeric(n_trials)
    
    for(t in 1:n_trials) {
        is_A <- rbinom(1, 1, probs[t])
        seq_idx[t] <- is_A
        
        p1 <- exp(beta*Q1)/(exp(beta*Q1) + exp(beta*Q2)); p2 <- exp(beta*Q2)/(exp(beta*Q1) + exp(beta*Q2))
        State_t[t] <- p1 - p2; Value_t[t] <- p1*Q1 + p2*Q2
        ent <- 0; if(p1>0) ent <- ent - p1*log2(p1); if(p2>0) ent <- ent - p2*log2(p2)
        Uncert_t[t] <- ent
        
        choice <- p_data$Resp[t]
        reward <- p_data$F[t]
        if(choice == 1) Q1 <- Q1 + alpha*(reward - Q1) else Q2 <- Q2 + alpha*(reward - Q2)
    }
    
    X_mat <- matrix(0, nrow=n_trials, ncol=128)
    for(t in 1:n_trials) { if(seq_idx[t] == 1) X_mat[t,] <- v_A else X_mat[t,] <- v_B }
    syn_Pi <- matrix(1, nrow=n_trials, ncol=128)
    
    sim <- sim_5layers_patched(n_trials, X_mat, ITI_sec, W_gen, W_ach1, W_ach2, W_thal, syn_Pi,
                       b_p[1], b_p[2], b_p[3], b_p[4], b_p[5], b_p[6], b_p[7], b_p[8], b_p[9])
    
    res_RL[[s_idx]] <- data.frame(pid = pid, nt = p_data$nt, State=State_t, Value=Value_t, Uncert=Uncert_t)
    res_W[[s_idx]] <- sim$W; res_D[[s_idx]] <- sim$D
}

df_RL <- do.call(rbind, res_RL)
mat_W <- do.call(rbind, res_W)
mat_D <- do.call(rbind, res_D)

layers <- list("Purkinje (W)" = mat_W, "DCN (D)" = mat_D)
targets <- c("State", "Value", "Uncertainty")

window_size <- 30
step_size <- 5
max_nt <- max(df_RL$nt, na.rm=TRUE)

results <- data.frame()
cat("Running Rolling Ridge Regression (Window Size =", window_size, ")...\n")

for(t_end in seq(window_size, max_nt, by = step_size)) {
    idx <- which(df_RL$nt > (t_end - window_size) & df_RL$nt <= t_end)
    if(length(idx) < 50) next 
    
    for(l_name in names(layers)) {
        X_full <- layers[[l_name]]
        X_win <- X_full[idx, , drop=FALSE]
        
        sd_cols <- apply(X_win, 2, sd)
        X_win <- X_win[, sd_cols > 1e-8, drop=FALSE]
        if(ncol(X_win) < 2) next
        
        for(t_name in targets) {
            # Map target name to column
            col_name <- if(t_name == "Uncertainty") "Uncert" else t_name
            Y_win <- df_RL[[col_name]][idx]
            
            fit <- tryCatch({
                cv.glmnet(X_win, Y_win, alpha=0, standardize=TRUE, nfolds=5)
            }, error = function(e) NULL)
            
            if(!is.null(fit)) {
                r2 <- max(0, 1 - (min(fit$cvm) / var(Y_win)))
                results <- rbind(results, data.frame(Trial_End=t_end, Layer=l_name, Encoding=t_name, R2=r2))
            }
        }
    }
}

p <- ggplot(results, aes(x=Trial_End, y=R2, color=Encoding)) +
    geom_line(linewidth=1.2) +
    geom_point(size=2) +
    facet_wrap(~Layer, ncol=1, scales="free_y") +
    scale_color_manual(values=c("State"="dodgerblue", "Value"="firebrick", "Uncertainty"="forestgreen")) +
    labs(title="Dynamic Rolling Ridge Regression (Window = 30 Trials)", 
         subtitle="Tracking the encoding strength of Purkinje and DCN layers over the experiment",
         y="Cross-Validated R-squared", x="Trial Number (Window End)") +
    theme_minimal(base_size=14)

ggsave("results/Fig55_Rolling_Ridge.png", plot=p, width=12, height=8, dpi=300)
cat("Done.\n")

