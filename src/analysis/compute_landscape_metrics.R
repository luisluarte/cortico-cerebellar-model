library(Matrix)

cat("Computing 7-Dimensional Information Geometry Metrics for the Biological Basin...\n")

df <- readRDS('final_model/results/factorial_fixed_10k_A.rds')
obj <- df[['bio_dist']][['traces']][[1]]  # 5000 x 4 x 7

# Flatten the 5000 iterations and 4 chains into a 20000 x 7 matrix
mat <- matrix(obj, ncol = 7)

# 1. Compute empirical covariance matrix (Sigma)
sigma <- cov(mat)

# 2. Invert to get the Precision Matrix (Empirical Hessian)
# By Bernstein-von Mises, Precision = Expected Fisher Information Matrix = Hessian of NLL
hessian <- solve(sigma)

# 3. Extract Eigenspectrum
eigs <- eigen(hessian)$values

# The eigenvalues are sorted from largest to smallest by default
lambda_max <- max(eigs)
lambda_min <- min(eigs)
kappa <- lambda_max / lambda_min

# Print results
cat(sprintf("Maximum Eigenvalue (Lipschitz Smoothness L): %.4f\n", lambda_max))
cat(sprintf("Minimum Eigenvalue: %.4f\n", lambda_min))
cat(sprintf("Condition Number (Kappa): %.4f\n", kappa))

# Save to a text file
out <- file("results/landscape_metrics.txt")
writeLines(c(
  "--- High-Dimensional Landscape Metrics (7-Parameter MCMC Posterior) ---",
  sprintf("Lipschitz Smoothness (L = max eigenvalue): %.4f", lambda_max),
  sprintf("Minimum Eigenvalue: %.4f", lambda_min),
  sprintf("Gradient Isotropy / Condition Number (Kappa): %.4f", kappa)
), out)
close(out)

cat("Successfully calculated landscape metrics.\n")
