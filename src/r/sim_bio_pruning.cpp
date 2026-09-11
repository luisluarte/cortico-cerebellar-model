#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat simulate_bio_pruning(int N_trials, arma::mat X, arma::vec ITI,
                       arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::mat Pi_mat,
                       double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                       double p_alpha_gran, double p_beta_gran, double p_sigma2_diff,
                       double p_gran_ratio, double p_dcn_ratio) {
    
    // Max capacity based on initialized matrices
    int max_gran = W_ach1.n_rows;
    int max_dcn = W_ach2.n_rows;
    
    // Pruned capacity based on hyper-parameters
    int active_gran = std::round(p_gran_ratio * max_gran);
    int active_dcn = std::round(p_dcn_ratio * max_dcn);
    
    // Bounds check to avoid 0-dimension crash
    if(active_gran < 2) active_gran = 2;
    if(active_dcn < 2) active_dcn = 2;
    
    arma::vec mu = arma::zeros(32);
    arma::vec Z = arma::zeros(active_gran);
    arma::vec W_purk = arma::zeros(active_gran);
    arma::vec D = arma::zeros(active_dcn);
    
    arma::mat mu_history(N_trials, 32, arma::fill::zeros);
    bool diverged = false;
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI[t] > 0) {
            Z.zeros();
            W_purk += arma::randn<arma::vec>(active_gran) * std::sqrt(p_sigma2_diff * ITI[t]);
        }
        
        arma::vec I_t = X.row(t).t();
        arma::vec I_hat = W_gen * mu;
        arma::vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        
        // Thalamic feedback from active DCN
        arma::vec thal_feedback = W_thal.head_cols(active_dcn) * D;
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * thal_feedback);
        
        if(!mu.is_finite() || arma::max(arma::abs(mu)) > 1e4) {
            diverged = true;
            break;
        }
        
        // Granule Projection (Pruned)
        arma::vec G = W_ach1.head_rows(active_gran) * mu;
        
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = arma::mean(arma::abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        
        // DCN Compression (Pruned)
        arma::mat W_ach2_active = W_ach2.submat(0, 0, active_dcn - 1, active_gran - 1);
        D = W_ach2_active * (G % W_purk);
        
        mu_history.row(t) = mu.t();
    }
    
    if(diverged) {
        arma::mat err(1,1); err.fill(NA_REAL);
        return err;
    }
    
    return mu_history;
}
