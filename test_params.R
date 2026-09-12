post_df <- readRDS('final_model/results/factorial_fixed_10k_A.rds')
params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) exp(apply(x, 3, mean))))
colnames(params) <- c('alpha_pc', 'lambda_pc', 'beta_thal', 'kappa_cf', 'alpha_gran', 'beta_gran', 'sigma2')
cormat <- cor(params, method='spearman')
print(round(cormat, 3))

for(i in 1:6) {
  for(j in (i+1):7) {
    if(abs(cormat[i,j]) > 0.3) {
      cat(sprintf("Found high correlation: %s vs %s (rho = %.3f)\n", colnames(cormat)[i], colnames(cormat)[j], cormat[i,j]))
    }
  }
}
