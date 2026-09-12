#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List sim_restless_agent(int N_trials, double sigma_drift,
                       mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, vec Pi_vec,
                       double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                       double p_alpha_gran, double p_beta_gran, double p_sigma2_diff,
                       double p_gran_ratio, double p_dcn_ratio,
                       vec W_readout, double intercept) {
    
    int dim_X = 27, dim_Z = 32;
    int active_gran = std::round(p_gran_ratio * W_ach1.n_rows);
    int active_dcn = std::round(p_dcn_ratio * W_ach2.n_rows);
    if(active_gran < 2) active_gran = 2;
    if(active_dcn < 2) active_dcn = 2;
    
    vec mu = zeros<vec>(dim_Z);
    vec Z = zeros<vec>(active_gran);
    vec W_purk = zeros<vec>(active_gran);
    vec D = zeros<vec>(active_dcn);
    
    mat sub_W_ach1 = W_ach1.rows(0, active_gran - 1);
    mat sub_W_ach2 = W_ach2.submat(0, 0, active_dcn - 1, active_gran - 1);
    mat sub_W_thal = W_thal.cols(0, active_dcn - 1);
    
    vec choices = zeros<vec>(N_trials);
    vec rewards = zeros<vec>(N_trials);
    
    // Initial probabilities
    double pA = 0.75;
    double pB = 0.25;
    
    mat X = zeros<mat>(N_trials, dim_X);
    for(int t=0; t<N_trials; t++) {
        X(t, 0) = 1; // Arm A is active
        X(t, 8) = 1; // Arm B is active
    }
    
    double ITI = 1.0; // Assume constant fast ITI for the task
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI > 0) {
            Z.zeros();
            W_purk += randn<vec>(active_gran) * std::sqrt(p_sigma2_diff * ITI);
        }
        
        vec I_t = X.row(t).t();
        vec I_hat = W_gen * mu;
        vec eps = Pi_vec % (I_t - I_hat);
        
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI) * mu + p_beta_thal * (sub_W_thal * D));
        
        vec G = sub_W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = mean(abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        
        vec P = G % W_purk;
        D = sub_W_ach2 * P;
        
        // Agent Decision
        double logit = dot(mu, W_readout) + intercept;
        double p_choose_1 = 1.0 / (1.0 + std::exp(-logit));
        int choice = R::runif(0.0, 1.0) < p_choose_1 ? 1 : 0;
        choices[t] = choice;
        
        // Environment evaluates the choice
        double prob = (choice == 0) ? pA : pB;
        int reward = R::runif(0.0, 1.0) < prob ? 1 : 0;
        rewards[t] = reward;
        
        // Update next input
        if (t + 1 < N_trials) {
            X(t+1, 16) = reward;
            X(t+1, 26) = choice;
        }
        
        // Random Walk Drift for next trial (The Restless Bandit Mechanism)
        pA += R::rnorm(0, sigma_drift);
        pB += R::rnorm(0, sigma_drift);
        
        // Reflecting Boundaries at 0.1 and 0.9 to prevent certainties
        if(pA < 0.1) { pA = 0.1 + (0.1 - pA); }
        if(pA > 0.9) { pA = 0.9 - (pA - 0.9); }
        if(pB < 0.1) { pB = 0.1 + (0.1 - pB); }
        if(pB > 0.9) { pB = 0.9 - (pB - 0.9); }
    }
    
    return List::create(Named("choices") = choices, Named("rewards") = rewards);
}
