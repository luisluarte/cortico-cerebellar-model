
#ifndef BIO_BPTT_HPP
#define BIO_BPTT_HPP

#include <stan/math.hpp>
#include <Eigen/Dense>
#include <vector>

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// BPTT Forward and Backward engine
inline double run_bptt(
    const std::vector<int>& y,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& iti,
    const std::vector<int>& subj_id,
    const Eigen::MatrixXd& W_gen,
    const Eigen::MatrixXd& W_ach1,
    const Eigen::MatrixXd& W_ach2,
    const Eigen::MatrixXd& W_thal,
    const Eigen::VectorXd& Pi_vec,
    const Eigen::VectorXd& alpha_pc,
    const Eigen::VectorXd& lambda_pc,
    const Eigen::VectorXd& beta_thal,
    const Eigen::VectorXd& kappa_cf,
    const Eigen::VectorXd& alpha_gran,
    const Eigen::VectorXd& beta_gran,
    const Eigen::VectorXd& intercept,
    const Eigen::MatrixXd& W_readout,
    Eigen::VectorXd& grad_alpha_pc,
    Eigen::VectorXd& grad_lambda_pc,
    Eigen::VectorXd& grad_beta_thal,
    Eigen::VectorXd& grad_kappa_cf,
    Eigen::VectorXd& grad_alpha_gran,
    Eigen::VectorXd& grad_beta_gran,
    Eigen::VectorXd& grad_intercept,
    Eigen::MatrixXd& grad_W_readout
) {
    int N = y.size();
    int K = alpha_pc.size();
    
    // Gradients init
    grad_alpha_pc = Eigen::VectorXd::Zero(K);
    grad_lambda_pc = Eigen::VectorXd::Zero(K);
    grad_beta_thal = Eigen::VectorXd::Zero(K);
    grad_kappa_cf = Eigen::VectorXd::Zero(K);
    grad_alpha_gran = Eigen::VectorXd::Zero(K);
    grad_beta_gran = Eigen::VectorXd::Zero(K);
    grad_intercept = Eigen::VectorXd::Zero(K);
    grad_W_readout = Eigen::MatrixXd::Zero(W_readout.rows(), W_readout.cols());

    // Cached states
    Eigen::MatrixXd mu_in = Eigen::MatrixXd::Zero(N, 32);
    Eigen::MatrixXd mu_out = Eigen::MatrixXd::Zero(N, 32);
    Eigen::MatrixXd U_cache = Eigen::MatrixXd::Zero(N, 32);
    Eigen::MatrixXd eps_cache = Eigen::MatrixXd::Zero(N, 27);
    Eigen::MatrixXd G_cache = Eigen::MatrixXd::Zero(N, 896);
    Eigen::MatrixXd Z_in = Eigen::MatrixXd::Zero(N, 896);
    Eigen::MatrixXd Z_out = Eigen::MatrixXd::Zero(N, 896);
    Eigen::MatrixXd Wp_in = Eigen::MatrixXd::Zero(N, 896);
    Eigen::MatrixXd Wp_out = Eigen::MatrixXd::Zero(N, 896);
    Eigen::MatrixXd D_in = Eigen::MatrixXd::Zero(N, 362);
    Eigen::MatrixXd D_out = Eigen::MatrixXd::Zero(N, 362);
    Eigen::VectorXd eps_mag_cache = Eigen::VectorXd::Zero(N);
    
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(32);
    Eigen::VectorXd Z = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd Wp = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd D = Eigen::VectorXd::Zero(362);

    int current_subj = -1;
    double log_lik = 0.0;
    
    Eigen::MatrixXd W_gen_t = W_gen.transpose();

    // FORWARD PASS
    for(int t = 0; t < N; t++) {
        if(subj_id[t] != current_subj) {
            mu.setZero(); Z.setZero(); Wp.setZero(); D.setZero();
            current_subj = subj_id[t];
        }
        
        mu_in.row(t) = mu.transpose();
        Z_in.row(t) = Z.transpose();
        Wp_in.row(t) = Wp.transpose();
        D_in.row(t) = D.transpose();
        
        int k = current_subj - 1;
        
        Eigen::VectorXd I_t = X.row(t).transpose();
        Eigen::VectorXd I_hat = W_gen * mu;
        Eigen::VectorXd eps = Pi_vec.cwiseProduct(I_t - I_hat);
        eps_cache.row(t) = eps.transpose();
        
        Eigen::VectorXd U = (W_gen_t * eps) - (lambda_pc[k] * iti[t]) * mu + beta_thal[k] * (W_thal * D);
        U_cache.row(t) = U.transpose();
        
        mu = mu + alpha_pc[k] * U;
        mu_out.row(t) = mu.transpose();
        
        Eigen::VectorXd G = W_ach1 * mu;
        G_cache.row(t) = G.transpose();
        
        Z = (1.0 - beta_gran[k]) * Z + alpha_gran[k] * G;
        Z_out.row(t) = Z.transpose();
        
        double eps_mag = eps.cwiseAbs().mean();
        eps_mag_cache[t] = eps_mag;
        
        Wp = Wp - kappa_cf[k] * (eps_mag * Z);
        Wp_out.row(t) = Wp.transpose();
        
        D = W_ach2 * (G.cwiseProduct(Wp));
        D_out.row(t) = D.transpose();
        
        double logit = intercept[k] + mu.dot(W_readout.col(k));
        double prob = sigmoid(logit);
        if(y[t] == 1) {
            log_lik += std::log(prob + 1e-12);
        } else {
            log_lik += std::log(1.0 - prob + 1e-12);
        }
    }

    // BACKWARD PASS (Adjoint)
    Eigen::VectorXd d_mu = Eigen::VectorXd::Zero(32);
    Eigen::VectorXd d_Z = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd d_Wp = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd d_D = Eigen::VectorXd::Zero(362);
    
    Eigen::MatrixXd W_ach1_t = W_ach1.transpose();
    Eigen::MatrixXd W_ach2_t = W_ach2.transpose();
    Eigen::MatrixXd W_thal_t = W_thal.transpose();

    for(int t = N - 1; t >= 0; t--) {
        int k = subj_id[t] - 1;
        
        // Zero out gradients if we cross a subject boundary going backwards
        if(t < N - 1 && subj_id[t] != subj_id[t+1]) {
            d_mu.setZero(); d_Z.setZero(); d_Wp.setZero(); d_D.setZero();
        }
        
        Eigen::VectorXd mu_out_t = mu_out.row(t).transpose();
        double logit = intercept[k] + mu_out_t.dot(W_readout.col(k));
        double prob = sigmoid(logit);
        double d_logit = y[t] - prob;
        
        grad_intercept[k] += d_logit;
        grad_W_readout.col(k) += d_logit * mu_out_t;
        d_mu += d_logit * W_readout.col(k);
        
        // D_out
        Eigen::VectorXd d_D_out = d_D;
        Eigen::VectorXd Wp_out_t = Wp_out.row(t).transpose();
        Eigen::VectorXd G_cache_t = G_cache.row(t).transpose();
        
        Eigen::VectorXd d_G_D = (W_ach2_t * d_D_out).cwiseProduct(Wp_out_t);
        Eigen::VectorXd d_Wp_in_D = (W_ach2_t * d_D_out).cwiseProduct(G_cache_t);
        d_Wp += d_Wp_in_D;
        d_D.setZero(); 
        
        // Wp_out
        Eigen::VectorXd d_Wp_out = d_Wp;
        Eigen::VectorXd Z_out_t = Z_out.row(t).transpose();
        grad_kappa_cf[k] += - d_Wp_out.dot(Z_out_t) * eps_mag_cache[t];
        double d_eps_mag = - kappa_cf[k] * d_Wp_out.dot(Z_out_t);
        d_Z += - kappa_cf[k] * eps_mag_cache[t] * d_Wp_out;
        
        // Z_out
        Eigen::VectorXd d_Z_out = d_Z;
        Eigen::VectorXd Z_in_t = Z_in.row(t).transpose();
        grad_beta_gran[k] += - d_Z_out.dot(Z_in_t);
        grad_alpha_gran[k] += d_Z_out.dot(G_cache_t);
        Eigen::VectorXd d_G_Z = alpha_gran[k] * d_Z_out;
        d_Z = (1.0 - beta_gran[k]) * d_Z_out; 
        
        // G
        Eigen::VectorXd d_G = d_G_D + d_G_Z;
        d_mu += W_ach1_t * d_G;
        
        // eps_mag
        Eigen::VectorXd d_eps = Eigen::VectorXd::Zero(27);
        for(int i = 0; i < 27; i++) {
            double sign_eps = eps_cache(t, i) > 0 ? 1.0 : (eps_cache(t, i) < 0 ? -1.0 : 0.0);
            d_eps[i] += d_eps_mag * (1.0 / 27.0) * sign_eps;
        }
        
        // mu_out
        Eigen::VectorXd d_mu_out = d_mu;
        Eigen::VectorXd U_cache_t = U_cache.row(t).transpose();
        grad_alpha_pc[k] += d_mu_out.dot(U_cache_t);
        Eigen::VectorXd d_U = alpha_pc[k] * d_mu_out;
        
        d_eps += W_gen * d_U;
        Eigen::VectorXd mu_in_t = mu_in.row(t).transpose();
        grad_lambda_pc[k] += - d_U.dot(mu_in_t) * iti[t];
        
        Eigen::VectorXd D_in_t = D_in.row(t).transpose();
        grad_beta_thal[k] += d_U.dot(W_thal * D_in_t);
        
        d_D += W_thal_t * (d_U * beta_thal[k]); 
        d_mu = d_mu_out - d_U * (lambda_pc[k] * iti[t]); 
        
        // eps
        d_mu += W_gen_t * (- Pi_vec.cwiseProduct(d_eps));
    }
    
    return log_lik;
}

namespace full_bio_bptt_model_namespace {

template <bool propto__, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16>
stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16>
bio_log_lpmf(const T0& y, const T1& X_arg, const T2& iti_arg, const T3& subj_id,
             const T4& W_gen_arg, const T5& W_ach1_arg, const T6& W_ach2_arg, const T7& W_thal_arg, const T8& Pi_vec_arg,
             const T9& alpha_pc_arg, const T10& lambda_pc_arg, const T11& beta_thal_arg, const T12& kappa_cf_arg,
             const T13& alpha_gran_arg, const T14& beta_gran_arg, const T15& intercept_arg, const T16& W_readout_arg,
             std::ostream* pstream__) {
    
    // Extract values to Eigen
    Eigen::MatrixXd X = stan::math::value_of(X_arg);
    Eigen::VectorXd iti = stan::math::value_of(iti_arg);
    Eigen::MatrixXd W_gen = stan::math::value_of(W_gen_arg);
    Eigen::MatrixXd W_ach1 = stan::math::value_of(W_ach1_arg);
    Eigen::MatrixXd W_ach2 = stan::math::value_of(W_ach2_arg);
    Eigen::MatrixXd W_thal = stan::math::value_of(W_thal_arg);
    Eigen::VectorXd Pi_vec = stan::math::value_of(Pi_vec_arg);
    
    Eigen::VectorXd alpha_pc = stan::math::value_of(alpha_pc_arg);
    Eigen::VectorXd lambda_pc = stan::math::value_of(lambda_pc_arg);
    Eigen::VectorXd beta_thal = stan::math::value_of(beta_thal_arg);
    Eigen::VectorXd kappa_cf = stan::math::value_of(kappa_cf_arg);
    Eigen::VectorXd alpha_gran = stan::math::value_of(alpha_gran_arg);
    Eigen::VectorXd beta_gran = stan::math::value_of(beta_gran_arg);
    Eigen::VectorXd intercept = stan::math::value_of(intercept_arg);
    Eigen::MatrixXd W_readout = stan::math::value_of(W_readout_arg);
    
    int K = alpha_pc.size();
    
    Eigen::VectorXd grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran, grad_intercept;
    Eigen::MatrixXd grad_W_readout;
    
    double log_lik = run_bptt(y, X, iti, subj_id, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                              alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_gran, beta_gran, intercept, W_readout,
                              grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran, grad_intercept, grad_W_readout);
    
    using return_t = stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16>;
    if constexpr (!stan::is_var<return_t>::value) {
        return log_lik;
    } else {
        return stan::math::make_callback_var(log_lik, [=](auto& vi) mutable {
            for(int k=0; k<K; k++) {
                stan::math::forward_as<stan::math::var>(alpha_pc_arg(k)).adj() += grad_alpha_pc(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(lambda_pc_arg(k)).adj() += grad_lambda_pc(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(beta_thal_arg(k)).adj() += grad_beta_thal(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(kappa_cf_arg(k)).adj() += grad_kappa_cf(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(alpha_gran_arg(k)).adj() += grad_alpha_gran(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(beta_gran_arg(k)).adj() += grad_beta_gran(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(intercept_arg(k)).adj() += grad_intercept(k) * vi.adj();
                for(int d=0; d<32; d++) {
                    stan::math::forward_as<stan::math::var>(W_readout_arg(d, k)).adj() += grad_W_readout(d, k) * vi.adj();
                }
            }
        });
    }
}

} // namespace

namespace full_bio_bptt_cp_model_namespace {

template <bool propto__, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16>
stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16>
bio_log_lpmf(const T0& y, const T1& X_arg, const T2& iti_arg, const T3& subj_id,
             const T4& W_gen_arg, const T5& W_ach1_arg, const T6& W_ach2_arg, const T7& W_thal_arg, const T8& Pi_vec_arg,
             const T9& alpha_pc_arg, const T10& lambda_pc_arg, const T11& beta_thal_arg, const T12& kappa_cf_arg,
             const T13& alpha_gran_arg, const T14& beta_gran_arg, const T15& intercept_arg, const T16& W_readout_arg,
             std::ostream* pstream__) {
    
    // Extract values to Eigen
    Eigen::MatrixXd X = stan::math::value_of(X_arg);
    Eigen::VectorXd iti = stan::math::value_of(iti_arg);
    Eigen::MatrixXd W_gen = stan::math::value_of(W_gen_arg);
    Eigen::MatrixXd W_ach1 = stan::math::value_of(W_ach1_arg);
    Eigen::MatrixXd W_ach2 = stan::math::value_of(W_ach2_arg);
    Eigen::MatrixXd W_thal = stan::math::value_of(W_thal_arg);
    Eigen::VectorXd Pi_vec = stan::math::value_of(Pi_vec_arg);
    
    Eigen::VectorXd alpha_pc = stan::math::value_of(alpha_pc_arg);
    Eigen::VectorXd lambda_pc = stan::math::value_of(lambda_pc_arg);
    Eigen::VectorXd beta_thal = stan::math::value_of(beta_thal_arg);
    Eigen::VectorXd kappa_cf = stan::math::value_of(kappa_cf_arg);
    Eigen::VectorXd alpha_gran = stan::math::value_of(alpha_gran_arg);
    Eigen::VectorXd beta_gran = stan::math::value_of(beta_gran_arg);
    Eigen::VectorXd intercept = stan::math::value_of(intercept_arg);
    Eigen::MatrixXd W_readout = stan::math::value_of(W_readout_arg);
    
    int K = alpha_pc.size();
    
    Eigen::VectorXd grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran, grad_intercept;
    Eigen::MatrixXd grad_W_readout;
    
    double log_lik = run_bptt(y, X, iti, subj_id, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                              alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_gran, beta_gran, intercept, W_readout,
                              grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran, grad_intercept, grad_W_readout);
    
    using return_t = stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16>;
    if constexpr (!stan::is_var<return_t>::value) {
        return log_lik;
    } else {
        return stan::math::make_callback_var(log_lik, [=](auto& vi) mutable {
            for(int k=0; k<K; k++) {
                stan::math::forward_as<stan::math::var>(alpha_pc_arg(k)).adj() += grad_alpha_pc(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(lambda_pc_arg(k)).adj() += grad_lambda_pc(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(beta_thal_arg(k)).adj() += grad_beta_thal(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(kappa_cf_arg(k)).adj() += grad_kappa_cf(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(alpha_gran_arg(k)).adj() += grad_alpha_gran(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(beta_gran_arg(k)).adj() += grad_beta_gran(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(intercept_arg(k)).adj() += grad_intercept(k) * vi.adj();
                for(int d=0; d<32; d++) {
                    stan::math::forward_as<stan::math::var>(W_readout_arg(d, k)).adj() += grad_W_readout(d, k) * vi.adj();
                }
            }
        });
    }
}

} // namespace
#endif
