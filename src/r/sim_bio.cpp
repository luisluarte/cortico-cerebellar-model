#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat simulate_bio(int N_trials, arma::mat X, arma::vec ITI,
                       arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::vec Pi_vec,
                       double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                       double p_alpha_gran, double p_beta_gran, double p_sigma2_diff) {
    
    arma::vec mu = arma::zeros(32);
    arma::vec Z = arma::zeros(896);
    arma::vec W_purk = arma::zeros(896);
    arma::vec D = arma::zeros(362);
    
    arma::mat mu_history(N_trials, 32, arma::fill::zeros);
    bool diverged = false;
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI[t] > 0) {
            Z.zeros();
            W_purk += arma::randn<arma::vec>(896) * std::sqrt(p_sigma2_diff * ITI[t]);
        }
        
        arma::vec I_t = X.row(t).t();
        arma::vec I_hat = W_gen * mu;
        arma::vec eps = Pi_vec % (I_t - I_hat);
        
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * (W_thal * D));
        
        if(!mu.is_finite() || arma::max(arma::abs(mu)) > 1e4) {
            diverged = true;
            break;
        }
        
        arma::vec G = W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = arma::mean(arma::abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        D = W_ach2 * (G % W_purk);
        
        mu_history.row(t) = mu.t();
    }
    
    if(diverged) {
        arma::mat err(1,1); err.fill(NA_REAL);
        return err;
    }
    
    return mu_history;
}
