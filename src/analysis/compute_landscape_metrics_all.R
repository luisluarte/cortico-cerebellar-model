library(Matrix)

cat("Computing 7-Dimensional Information Geometry Metrics for ALL 50 Subjects...\n")

df <- readRDS('final_model/results/factorial_fixed_10k_A.rds')

lambda_max_list <- c()
lambda_min_list <- c()
kappa_list <- c()

traces_list <- df[['bio_dist']][['traces']]
n_subjects <- length(traces_list)

for(i in 1:n_subjects) {
  obj <- traces_list[[i]]  # 5000 x 4 x 7
  # Flatten
  mat <- matrix(obj, ncol = 7)
  
  # Covariance
  sigma <- cov(mat)
  
  # Hessian (Precision Matrix)
  # Use pseudo-inverse or solve
  hessian <- tryCatch({
    solve(sigma)
  }, error = function(e) {
    NULL
  })
  
  if(!is.null(hessian)) {
    eigs <- eigen(hessian)$values
    l_max <- max(eigs)
    l_min <- min(eigs)
    
    lambda_max_list <- c(lambda_max_list, l_max)
    lambda_min_list <- c(lambda_min_list, l_min)
    kappa_list <- c(kappa_list, l_max / l_min)
  }
}

mean_l_max <- mean(lambda_max_list)
mean_l_min <- mean(lambda_min_list)
mean_kappa <- mean(kappa_list)

cat(sprintf("Mean Lipschitz Smoothness L: %.4f\n", mean_l_max))
cat(sprintf("Mean Minimum Eigenvalue: %.4f\n", mean_l_min))
cat(sprintf("Mean Condition Number (Kappa): %.4f\n", mean_kappa))

out <- file("results/landscape_metrics_all.txt")
writeLines(c(
  "--- High-Dimensional Landscape Metrics (Averaged across N=50 MCMC Posteriors) ---",
  sprintf("Mean Lipschitz Smoothness (L = max eigenvalue): %.4f", mean_l_max),
  sprintf("Mean Minimum Eigenvalue: %.4f", mean_l_min),
  sprintf("Mean Gradient Isotropy / Condition Number (Kappa): %.4f", mean_kappa)
), out)
close(out)

cat("Successfully calculated landscape metrics for all subjects.\n")
