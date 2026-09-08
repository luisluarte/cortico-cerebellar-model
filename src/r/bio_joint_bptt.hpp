
#ifndef BIO_JOINT_BPTT_HPP
#define BIO_JOINT_BPTT_HPP

#include <stan/math.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double run_bptt_joint(
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
    const Eigen::VectorXd& intercept_c,
    const Eigen::MatrixXd& W_c,
    const Eigen::VectorXd& RT,
    const Eigen::VectorXd& intercept_rt,
    const Eigen::MatrixXd& W_rt,
    const Eigen::VectorXd& sigma_rt,
    const Eigen::VectorXd& tau_rt,
    Eigen::VectorXd& grad_alpha_pc,
    Eigen::VectorXd& grad_lambda_pc,
    Eigen::VectorXd& grad_beta_thal,
    Eigen::VectorXd& grad_kappa_cf,
    Eigen::VectorXd& grad_alpha_gran,
    Eigen::VectorXd& grad_beta_gran,
    Eigen::VectorXd& grad_intercept_c,
    Eigen::MatrixXd& grad_W_c,
    Eigen::VectorXd& grad_intercept_rt,
    Eigen::MatrixXd& grad_W_rt,
    Eigen::VectorXd& grad_sigma_rt,
    Eigen::VectorXd& grad_tau_rt
) {
    int N = y.size();
    
    Eigen::MatrixXd W_ach1_t = W_ach1.transpose();
    Eigen::MatrixXd W_ach2_t = W_ach2.transpose();
    Eigen::SparseMatrix<double> W_ach1_s = W_ach1.sparseView();
    Eigen::SparseMatrix<double> W_ach2_s = W_ach2.sparseView();
    Eigen::SparseMatrix<double> W_ach1_t_s = W_ach1_t.sparseView();
    Eigen::SparseMatrix<double> W_ach2_t_s = W_ach2_t.sparseView();

    int K = alpha_pc.size();
    
    // Gradients init
    grad_alpha_pc = Eigen::VectorXd::Zero(K);
    grad_lambda_pc = Eigen::VectorXd::Zero(K);
    grad_beta_thal = Eigen::VectorXd::Zero(K);
    grad_kappa_cf = Eigen::VectorXd::Zero(K);
    grad_alpha_gran = Eigen::VectorXd::Zero(K);
    grad_beta_gran = Eigen::VectorXd::Zero(K);
    grad_intercept_c = Eigen::VectorXd::Zero(K);
    grad_W_c = Eigen::MatrixXd::Zero(W_c.rows(), W_c.cols());
    
    grad_intercept_rt = Eigen::VectorXd::Zero(K);
    grad_W_rt = Eigen::MatrixXd::Zero(W_rt.rows(), W_rt.cols());
    grad_sigma_rt = Eigen::VectorXd::Zero(K);
    grad_tau_rt = Eigen::VectorXd::Zero(K);

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
        
        Eigen::VectorXd G = W_ach1_s * mu;
        G_cache.row(t) = G.transpose();
        
        Z = (1.0 - beta_gran[k]) * Z + alpha_gran[k] * G;
        Z_out.row(t) = Z.transpose();
        
        double eps_mag = eps.cwiseAbs().mean();
        eps_mag_cache[t] = eps_mag;
        
        Wp = Wp - kappa_cf[k] * (eps_mag * Z);
        Wp_out.row(t) = Wp.transpose();
        
        D = W_ach2_s * (G.cwiseProduct(Wp));
        D_out.row(t) = D.transpose();
        
        // Choice
        double logit = intercept_c[k] + mu.dot(W_c.col(k));
        double prob = sigmoid(logit);
        if(y[t] == 1) {
            log_lik += std::log(prob + 1e-12);
        } else {
            log_lik += std::log(1.0 - prob + 1e-12);
        }
        
        // RT (Robust Student-t, nu=4)
        double rt_pred = intercept_rt[k] + mu.dot(W_rt.col(k));
        double rt_shifted = RT[t] - tau_rt[k];
        double diff = std::log(rt_shifted + 1e-12) - rt_pred;
        double sigma_sq = sigma_rt[k] * sigma_rt[k];
        double nu = 4.0;
        double u = (diff * diff) / (nu * sigma_sq);
        
        double current_ll = -std::log(rt_shifted + 1e-12) - std::log(sigma_rt[k] + 1e-12) 
                            - 0.5 * (nu + 1.0) * std::log(1.0 + u);
        if (std::isnan(current_ll) || std::isinf(current_ll)) {
            std::cout << "NAN LL! diff=" << diff << " rt_shifted=" << rt_shifted << " rt_pred=" << rt_pred << " u=" << u << std::endl;
        }
        log_lik += current_ll;
    }
    
    static int eval_count = 0;
    eval_count++;
    if(eval_count <= 5 || std::isnan(log_lik)) {
        std::cout << "Eval " << eval_count << " final log_lik = " << log_lik << std::endl;
    }
    
    // BACKWARD PASS
    Eigen::MatrixXd W_thal_t = W_thal.transpose();

    
    Eigen::VectorXd d_mu = Eigen::VectorXd::Zero(32);
    Eigen::VectorXd d_Z = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd d_Wp = Eigen::VectorXd::Zero(896);
    Eigen::VectorXd d_D = Eigen::VectorXd::Zero(362);

    for(int t = N - 1; t >= 0; t--) {
        int k = subj_id[t] - 1;
        
        // --- JOINT GRADIENTS ---
        double logit = intercept_c[k] + mu_out.row(t).dot(W_c.col(k));
        double prob = sigmoid(logit);
        double d_LL_logit = y[t] - prob;
        
        double rt_pred = intercept_rt[k] + mu_out.row(t).dot(W_rt.col(k));
        double rt_shifted = RT[t] - tau_rt[k];
        double diff = std::log(rt_shifted + 1e-12) - rt_pred;
        double sigma_sq = sigma_rt[k] * sigma_rt[k];
        double nu = 4.0;
        double u = (diff * diff) / (nu * sigma_sq);
        
        // Student-t gradients
        double d_LL_rt_pred = ((nu + 1.0) / (nu * sigma_sq)) * diff / (1.0 + u);
        double d_LL_tau_rt = (1.0 / (rt_shifted + 1e-12)) * (1.0 + d_LL_rt_pred);
        double d_LL_sigma_rt = -(1.0 / (sigma_rt[k] + 1e-12)) + ((nu + 1.0) * u) / (sigma_rt[k] * (1.0 + u));
        
        grad_intercept_c[k] += d_LL_logit;
        grad_W_c.col(k) += d_LL_logit * mu_out.row(t).transpose();
        
        grad_intercept_rt[k] += d_LL_rt_pred;
        grad_W_rt.col(k) += d_LL_rt_pred * mu_out.row(t).transpose();
        
        grad_sigma_rt[k] += d_LL_sigma_rt;
        grad_tau_rt[k] += d_LL_tau_rt;
        
        Eigen::VectorXd d_mu_out = d_mu + d_LL_logit * W_c.col(k) + d_LL_rt_pred * W_rt.col(k);
        // -----------------------
        
        Eigen::VectorXd G = G_cache.row(t).transpose();
        Eigen::VectorXd Z_prev = Z_in.row(t).transpose();
        Eigen::VectorXd Wp_prev = Wp_in.row(t).transpose();
        double eps_mag = eps_mag_cache[t];
        
        Eigen::VectorXd d_G = W_ach2_t_s * d_D; 
        d_G = d_G.cwiseProduct(Wp_prev);
        
        d_Wp += W_ach2_t_s * d_D;
        d_Wp = d_Wp.cwiseProduct(G);
        
        grad_kappa_cf[k] += d_Wp.dot(-eps_mag * Z_prev);
        d_Z += d_Wp * (-kappa_cf[k] * eps_mag);
        
        grad_alpha_gran[k] += d_Z.dot(G);
        d_G += d_Z * alpha_gran[k];
        
        grad_beta_gran[k] += d_Z.dot(-Z_prev);
        d_Z = d_Z * (1.0 - beta_gran[k]); 
        
        d_mu_out += W_ach1_t_s * d_G;
        
        Eigen::VectorXd U = U_cache.row(t).transpose();
        grad_alpha_pc[k] += d_mu_out.dot(U);
        Eigen::VectorXd d_U = alpha_pc[k] * d_mu_out;
        
        Eigen::VectorXd eps = eps_cache.row(t).transpose();
        Eigen::VectorXd d_eps = d_U.transpose() * W_gen_t; 
        
        Eigen::VectorXd d_eps_mag = d_Wp.cwiseProduct(-kappa_cf[k] * Z_prev);
        double sum_d_eps_mag = d_eps_mag.sum();
        
        for(int i=0; i<27; i++) {
            if(eps(i) > 0) d_eps(i) += sum_d_eps_mag / 27.0;
            else if(eps(i) < 0) d_eps(i) -= sum_d_eps_mag / 27.0;
        }
        
        Eigen::VectorXd mu_in_t = mu_in.row(t).transpose();
        grad_lambda_pc[k] += - d_U.dot(mu_in_t) * iti[t];
        
        Eigen::VectorXd D_in_t = D_in.row(t).transpose();
        grad_beta_thal[k] += d_U.dot(W_thal * D_in_t);
        
        d_D += W_thal_t * (d_U * beta_thal[k]); 
        d_mu = d_mu_out - d_U * (lambda_pc[k] * iti[t]); 
        
        d_mu += W_gen_t * (- Pi_vec.cwiseProduct(d_eps));
        
        if(t > 0 && subj_id[t] != subj_id[t-1]) {
            d_mu.setZero(); d_Z.setZero(); d_Wp.setZero(); d_D.setZero();
        }
    }
    
    return log_lik;
}

namespace full_bio_joint_model_namespace {

template <bool propto__, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21>
stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16, T18, T19, T20, T21>
bio_joint_log_lpmf(const T0& y, const T1& X_arg, const T2& iti_arg, const T3& subj_id,
             const T4& W_gen_arg, const T5& W_ach1_arg, const T6& W_ach2_arg, const T7& W_thal_arg, const T8& Pi_vec_arg,
             const T9& alpha_pc_arg, const T10& lambda_pc_arg, const T11& beta_thal_arg, const T12& kappa_cf_arg,
             const T13& alpha_gran_arg, const T14& beta_gran_arg, 
             const T15& intercept_c_arg, const T16& W_c_arg,
             const T17& RT_arg, const T18& intercept_rt_arg, const T19& W_rt_arg, const T20& sigma_rt_arg, const T21& tau_rt_arg,
             std::ostream* pstream__) {
    
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
    
    Eigen::VectorXd intercept_c = stan::math::value_of(intercept_c_arg);
    Eigen::MatrixXd W_c = stan::math::value_of(W_c_arg);
    
    Eigen::VectorXd RT = stan::math::value_of(RT_arg);
    Eigen::VectorXd intercept_rt = stan::math::value_of(intercept_rt_arg);
    Eigen::MatrixXd W_rt = stan::math::value_of(W_rt_arg);
    Eigen::VectorXd sigma_rt = stan::math::value_of(sigma_rt_arg);
    Eigen::VectorXd tau_rt = stan::math::value_of(tau_rt_arg);
    
    int K = alpha_pc.size();
    
    Eigen::VectorXd grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran;
    Eigen::VectorXd grad_intercept_c, grad_intercept_rt, grad_sigma_rt, grad_tau_rt;
    Eigen::MatrixXd grad_W_c, grad_W_rt;
    
    double log_lik = run_bptt_joint(y, X, iti, subj_id, W_gen, W_ach1, W_ach2, W_thal, Pi_vec,
                                    alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_gran, beta_gran,
                                    intercept_c, W_c, RT, intercept_rt, W_rt, sigma_rt, tau_rt,
                                    grad_alpha_pc, grad_lambda_pc, grad_beta_thal, grad_kappa_cf, grad_alpha_gran, grad_beta_gran,
                                    grad_intercept_c, grad_W_c, grad_intercept_rt, grad_W_rt, grad_sigma_rt, grad_tau_rt);
    
    using return_t = stan::return_type_t<T9, T10, T11, T12, T13, T14, T15, T16, T18, T19, T20, T21>;
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
                
                stan::math::forward_as<stan::math::var>(intercept_c_arg(k)).adj() += grad_intercept_c(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(intercept_rt_arg(k)).adj() += grad_intercept_rt(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(sigma_rt_arg(k)).adj() += grad_sigma_rt(k) * vi.adj();
                stan::math::forward_as<stan::math::var>(tau_rt_arg(k)).adj() += grad_tau_rt(k) * vi.adj();
                
                for(int d=0; d<32; d++) {
                    stan::math::forward_as<stan::math::var>(W_c_arg(d, k)).adj() += grad_W_c(d, k) * vi.adj();
                    stan::math::forward_as<stan::math::var>(W_rt_arg(d, k)).adj() += grad_W_rt(d, k) * vi.adj();
                }
            }
        });
    }
}

} // namespace
#endif
