#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List simulate_hallucination(int cutoff_trial, int hallucinate_steps, double hallucinate_ITI,
                            arma::mat X, arma::vec ITI,
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
    
    arma::mat sub_W_ach1 = W_ach1.rows(0, active_gran - 1);
    arma::mat sub_W_ach2 = W_ach2.submat(0, 0, active_dcn - 1, active_gran - 1);
    arma::mat sub_W_thal = W_thal.cols(0, active_dcn - 1);
    
    // 1. Standard task phase (up to cutoff)
    for(int t = 0; t <= cutoff_trial; t++) {
        if(ITI[t] > 0) {
            Z.zeros();
            W_purk += arma::randn<arma::vec>(active_gran) * std::sqrt(p_sigma2_diff * ITI[t]);
        }
        
        arma::vec I_t = X.row(t).t();
        arma::vec I_hat = W_gen * mu;
        arma::vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI[t]) * mu + p_beta_thal * (sub_W_thal * D));
        
        arma::vec G = sub_W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = arma::mean(arma::abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        
        arma::vec P = G % W_purk;
        D = sub_W_ach2 * P;
    }
    
    // 2. Hallucination (Sensory Deprivation) Phase
    arma::mat h_mu(hallucinate_steps, 32, arma::fill::zeros);
    arma::mat h_P(hallucinate_steps, active_gran, arma::fill::zeros);
    arma::mat h_I(hallucinate_steps, X.n_cols, arma::fill::zeros);
    
    for(int t = 0; t < hallucinate_steps; t++) {
        // Autonomous Rest: Purkinje weights undergo stochastic diffusion over the specified ITI
        W_purk += arma::randn<arma::vec>(active_gran) * std::sqrt(p_sigma2_diff * hallucinate_ITI);
        
        // Cut off external world (eps = 0). 
        // mu evolves autonomously via decay and internal DCN feedback over the ITI
        mu = mu + p_alpha_pc * ( - (p_lambda_pc * hallucinate_ITI) * mu + p_beta_thal * (sub_W_thal * D) );
        
        arma::vec G = sub_W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        arma::vec P = G % W_purk;
        D = sub_W_ach2 * P;
        
        // Generative hallucination of input
        arma::vec I_hat = W_gen * mu;
        
        h_mu.row(t) = mu.t();
        h_P.row(t) = P.t();
        h_I.row(t) = I_hat.t();
    }
    
    return List::create(Named("mu_hallucinated") = h_mu,
                        Named("P_hallucinated") = h_P,
                        Named("I_hallucinated") = h_I);
}
