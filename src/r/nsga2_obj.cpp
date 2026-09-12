
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

const vec L_bounds = {0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
const vec U_bounds = {1.500, 2.000, 1.500, 2.500, 1.500, 1.000, 5.000};

vec inv_logit_scaled(vec z) {
  return L_bounds + (U_bounds - L_bounds) % (1.0 / (1.0 + exp(-z)));
}

mat simulate_bio(int N_trials, mat X, vec ITI, vec reward,
                 mat W_gen, mat W_ach1, mat W_ach2, mat W_thal, mat Pi_mat,
                 vec params) {
    double p_alpha_pc   = params(0);
    double p_lambda_pc  = params(1);
    double p_beta_thal  = params(2);
    double p_kappa_cf   = params(3);
    double p_alpha_gran = params(4);
    double p_beta_gran  = params(5);
    double p_sigma2     = params(6);
    
    vec mu = zeros<vec>(32);
    vec Z = zeros<vec>(896);
    vec W_purk = zeros<vec>(896);
    vec D = zeros<vec>(362);
    
    mat mu_history = zeros<mat>(N_trials, 32);
    bool diverged = false;
    
    for(int t = 0; t < N_trials; t++) {
        if(ITI(t) > 0) {
            Z.zeros();
            W_purk += randn<vec>(896) * sqrt(p_sigma2 * ITI(t));
        }
        
        vec I_t = X.row(t).t();
        vec I_hat = W_gen * mu; 
        vec eps = Pi_mat.row(t).t() % (I_t - I_hat);
        
        mu = mu + p_alpha_pc * ((W_gen.t() * eps) - (p_lambda_pc * ITI(t)) * mu + p_beta_thal * (W_thal * D)); 
        
        if(!mu.is_finite() || max(abs(mu)) > 1e4) {
            diverged = true;
            break;
        }
        
        vec G = W_ach1 * mu;
        Z = (1.0 - p_beta_gran) * Z + p_alpha_gran * G;
        double eps_mag = mean(abs(eps));
        W_purk = W_purk - p_kappa_cf * (eps_mag * Z);
        D = W_ach2 * (G % W_purk);
        
        mu_history.row(t) = mu.t();
    }
    
    if(diverged) {
        mat err(1,1); err.fill(NA_REAL);
        return err;
    }
    return mu_history;
}

double eval_distillation(mat bio_mu, mat rnn_mu) {
    if(bio_mu.n_cols == 1 && !bio_mu.is_finite()) return -1e9;
    mat XX = bio_mu.t() * bio_mu;
    XX.diag() += 0.01;
    mat W;
    bool ok = solve(W, XX, bio_mu.t() * rnn_mu);
    if(!ok) return -1e9;
    mat proj = bio_mu * W;
    mat res = rnn_mu - proj;
    double sigma2 = accu(square(res)) / res.n_elem + 1e-4;
    return accu(-0.5 * log(2.0 * M_PI * sigma2) - sum(square(res), 1) / (2.0 * sigma2));
}

double eval_empirical(mat bio_mu, vec y_choice, vec y_rt) {
    if(bio_mu.n_cols == 1 && !bio_mu.is_finite()) return -1e9;
    
    mat XX = bio_mu.t() * bio_mu;
    XX.diag() += 0.01;
    vec W_c;
    bool ok_c = solve(W_c, XX, bio_mu.t() * (y_choice - 0.5)); 
    if(!ok_c) return -1e9;
    vec logits = bio_mu * W_c;
    vec p = 1.0 / (1.0 + exp(-logits));
    p = clamp(p, 1e-7, 1.0 - 1e-7);
    double bce = accu(y_choice % log(p) + (1.0 - y_choice) % log(1.0 - p));
    
    vec W_rt;
    bool ok_rt = solve(W_rt, XX, bio_mu.t() * y_rt);
    if(!ok_rt) return -1e9;
    vec bio_rt = bio_mu * W_rt;
    vec res = y_rt - bio_rt;
    double sigma_rt = sqrt(mean(square(res))) + 1e-4;
    double rt_ll = accu(-0.5 * log(2.0 * M_PI * sigma_rt * sigma_rt) - square(res) / (2.0 * sigma_rt * sigma_rt));
    
    return bce + rt_ll;
}

// [[Rcpp::export]]
NumericVector eval_nsga2_obj(NumericVector params_scaled_r, int N_subjs, NumericVector subj_indices_r, 
                             NumericMatrix X_r, NumericVector ITI_r, NumericVector reward_r, 
                             NumericVector choices_r, NumericVector y_rt_r, NumericMatrix rnn_mu_r, double gamma,
                             NumericMatrix W_gen_r, NumericMatrix W_ach1_r, NumericMatrix W_ach2_r, NumericMatrix W_thal_r, NumericMatrix Pi_mat_r) {
                             
    vec params = as<vec>(params_scaled_r);
    vec subj_indices = as<vec>(subj_indices_r);
    mat X = as<mat>(X_r);
    vec ITI = as<vec>(ITI_r);
    vec reward = as<vec>(reward_r);
    vec choices = as<vec>(choices_r);
    vec y_rt = as<vec>(y_rt_r);
    mat rnn_mu = as<mat>(rnn_mu_r);
    
    mat W_gen = as<mat>(W_gen_r);
    mat W_ach1 = as<mat>(W_ach1_r);
    mat W_ach2 = as<mat>(W_ach2_r);
    mat W_thal = as<mat>(W_thal_r);
    mat Pi_mat = as<mat>(Pi_mat_r);
    
    double total_nll = 0;
    double total_auc = 0; // Keeping interface, but we will just return NLL and gamma
    
    for(int s = 0; s < N_subjs; s++) {
        uvec idx = find(subj_indices == s);
        mat bio_mu = simulate_bio(idx.n_elem, X.rows(idx), ITI.elem(idx), reward.elem(idx), W_gen, W_ach1, W_ach2, W_thal, Pi_mat.rows(idx), params);
        
        double empirical_ll = eval_empirical(bio_mu, choices.elem(idx), y_rt.elem(idx));
        double dist_ll = eval_distillation(bio_mu, rnn_mu.rows(idx));
        
        // Blend likelihoods
        // Empirical LL is ~ -100 per subject. Distillation is ~ 1000 per subject.
        // We will just return -empirical_ll as objective 1, and -gamma as objective 2.
        // Wait, NSGA2 minimizes both objectives.
        // Objective 1: blended NLL = (1 - gamma) * -dist_ll + gamma * -empirical_ll
        // Objective 2: -gamma (to maximize gamma)
        
        total_nll += (1.0 - gamma) * (-dist_ll) + gamma * (-empirical_ll);
    }
    
    return NumericVector::create(total_nll, -gamma);
}
