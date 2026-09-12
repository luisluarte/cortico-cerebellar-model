
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat simulate_bio_gamma(int N_trials, arma::mat X, arma::vec ITI, arma::vec reward,
                             double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                             double p_alpha_gran, double p_beta_gran, double p_sigma2_diff) {
                             
    arma::vec mu = arma::zeros(32);
    arma::vec Z = arma::zeros(32); // Using simplified 32-D granular layer for speed/consistency with targets
    arma::mat W_purk = arma::zeros(32, 32);
    
    arma::mat mu_history(N_trials, 32, arma::fill::zeros);
    
    for(int t = 0; t < N_trials; t++) {
        // Stimulus mapping (X has 2 columns: bd1_scaled, bd2_scaled)
        arma::vec stim = arma::zeros(32);
        stim(0) = X(t, 0);
        stim(1) = X(t, 1);
        
        // Granular update
        Z = p_alpha_gran * Z + p_beta_gran * stim;
        
        // Purkinje update
        arma::vec purk = W_purk * Z;
        
        // Thalamic integration
        mu = p_alpha_pc * mu + p_beta_thal * purk;
        
        // Safety bounds
        if(!mu.is_finite() || arma::max(arma::abs(mu)) > 1e3) {
            mu.fill(1e3);
            mu_history.row(t) = mu.t();
            continue;
        }
        
        mu_history.row(t) = mu.t();
        
        // CF teaching signal (lag_reward)
        double r = reward[t];
        double cf_fire = 1.0 / (1.0 + exp(-r * p_kappa_cf));
        
        // Plasticity
        W_purk += p_lambda_pc * cf_fire * (Z * mu.t());
        
        // ITI decay
        double dt = ITI[t];
        Z *= exp(-dt * 0.1);
        mu *= exp(-dt * p_sigma2_diff);
    }
    
    return mu_history;
}
