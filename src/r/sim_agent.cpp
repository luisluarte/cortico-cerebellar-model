#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List simulate_agent_behavior(int N_trials, arma::mat X_base, arma::vec ITI,
                       arma::mat W_gen, arma::mat W_ach1, arma::mat W_ach2, arma::mat W_thal, arma::mat Pi_mat,
                       double p_alpha_pc, double p_lambda_pc, double p_beta_thal, double p_kappa_cf,
                       double p_alpha_gran, double p_beta_gran, double p_sigma2_diff,
                       double p_gran_ratio, double p_dcn_ratio,
                       arma::vec W_readout, double intercept, arma::vec arm_probs) {
    
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
    
    arma::vec choices = arma::zeros(N_trials);
    arma::vec rewards = arma::zeros(N_trials);
    
    arma::mat sub_W_ach1 = W_ach1.rows(0, active_gran - 1);
    arma::mat sub_W_ach2 = W_ach2.submat(0, 0, active_dcn - 1, active_gran - 1);
    arma::mat sub_W_thal = W_thal.cols(0, active_dcn - 1);
    
    arma::mat X = X_base;
    
    for(int t = 0; t < N_trials; t++) {
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
        
        // Agent makes a choice using logistic readout
        double logit = arma::dot(mu, W_readout) + intercept;
        double p_choose_1 = 1.0 / (1.0 + std::exp(-logit));
        int choice = R::runif(0.0, 1.0) < p_choose_1 ? 1 : 0;
        choices[t] = choice;
        
        // Environment evaluates the choice
        int armA = 0; int armB = 1;
        for(int i=0; i<8; i++) if(I_t[i] > 0.5) armA = i;
        for(int i=0; i<8; i++) if(I_t[i+8] > 0.5) armB = i;
        
        int chosen_arm = (choice == 0) ? armA : armB;
        double prob = arm_probs[chosen_arm];
        int reward = R::runif(0.0, 1.0) < prob ? 1 : 0;
        rewards[t] = reward;
        
        // Update next trial's input observation with consequence of this action
        if (t + 1 < N_trials) {
            X(t+1, 16) = reward; // lag_Reward index
            X(t+1, 26) = choice; // lag_Resp index
        }
    }
    
    return List::create(Named("choices") = choices,
                        Named("rewards") = rewards);
}
