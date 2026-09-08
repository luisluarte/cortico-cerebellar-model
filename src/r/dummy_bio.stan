
functions {
  real bio_log_lpmf(array[] int y, data matrix X, data vector iti, array[] int subj_id,
                    data matrix W_gen, data matrix W_ach1, data matrix W_ach2, data matrix W_thal, data vector Pi_vec,
                    vector alpha_pc, vector lambda_pc, vector beta_thal, vector kappa_cf,
                    vector alpha_gran, vector beta_gran, vector intercept, matrix W_readout) {
    return 0.0;
  }
}
data {
  int N;
}
