
if(file.exists("results/factorial_fixed_10k_checkpoint.rds")) {
  res <- readRDS("results/factorial_fixed_10k_checkpoint.rds")
  
  if("bio_dist" %in% names(res)) {
    cat("==== bio_dist RESULTS ====\n")
    
    # 1. Check LOO ELPD
    cat("ELPD (LOO): ", res$bio_dist$loo$elpd_loo, "\n")
    
    # Check if there are any -1e9 values
    ptw <- res$bio_dist$loo$pointwise[, "elpd_loo"]
    cat("Number of trials with ELPD < -1e6: ", sum(ptw < -1e6), "\n")
    
    # 2. Check Convergence
    cat("\nConvergence (Max Rhat): ", max(res$bio_dist$rhats, na.rm=TRUE), "\n")
    cat("Effective Sample Size (Min ESS): ", min(res$bio_dist$ess, na.rm=TRUE), "\n")
    
    cat("\n==== LOO Summary ====\n")
    print(res$bio_dist$loo)
  } else {
    cat("bio_dist not found in checkpoint.\n")
  }
} else {
  cat("Checkpoint file not found.\n")
}

