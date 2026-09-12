library(Matrix)
library(dplyr)
library(tidyr)

cat("Computing participant-level metrics for Distilled vs Empirical (Locked) Models...\n")

df <- readRDS('final_model/results/factorial_fixed_10k_A.rds')

traces_dist <- df[['bio_dist']][['traces']]
traces_emp <- df[['bio_lock']][['traces']]

n_subjects <- length(traces_dist)

results <- data.frame(
  Subject = integer(),
  Model = character(),
  L_max = numeric(),
  Kappa = numeric(),
  stringsAsFactors = FALSE
)

# Function to compute metrics from traces
get_metrics <- function(obj) {
  mat <- matrix(obj, ncol = 7)
  sigma <- cov(mat)
  hessian <- tryCatch({ solve(sigma) }, error = function(e) NULL)
  
  if(!is.null(hessian)) {
    eigs <- eigen(hessian)$values
    l_max <- max(eigs)
    l_min <- min(eigs)
    return(c(L_max = l_max, Kappa = l_max / l_min))
  } else {
    return(c(L_max = NA, Kappa = NA))
  }
}

for(i in 1:n_subjects) {
  # Distilled
  m_dist <- get_metrics(traces_dist[[i]])
  results <- rbind(results, data.frame(
    Subject = i,
    Model = "Distilled",
    L_max = m_dist["L_max"],
    Kappa = m_dist["Kappa"]
  ))
  
  # Empirical
  m_emp <- get_metrics(traces_emp[[i]])
  results <- rbind(results, data.frame(
    Subject = i,
    Model = "Empirical",
    L_max = m_emp["L_max"],
    Kappa = m_emp["Kappa"]
  ))
}

# Run paired statistics
dist_df <- results %>% filter(Model == "Distilled") %>% arrange(Subject)
emp_df <- results %>% filter(Model == "Empirical") %>% arrange(Subject)

# T-tests
t_L <- t.test(dist_df$L_max, emp_df$L_max, paired = TRUE)
t_K <- t.test(dist_df$Kappa, emp_df$Kappa, paired = TRUE)

cat("\n--- PAIRED T-TEST RESULTS (N=50) ---\n")
cat(sprintf("Lipschitz Smoothness (L_max):\n  Distilled Mean: %.4f\n  Empirical Mean: %.4f\n  p-value: %.2e\n", 
            mean(dist_df$L_max, na.rm=TRUE), mean(emp_df$L_max, na.rm=TRUE), t_L$p.value))

cat(sprintf("\nCondition Number (Kappa):\n  Distilled Mean: %.4f\n  Empirical Mean: %.4f\n  p-value: %.2e\n", 
            mean(dist_df$Kappa, na.rm=TRUE), mean(emp_df$Kappa, na.rm=TRUE), t_K$p.value))

# Save results
saveRDS(results, "results/landscape_comparison_stats.rds")
write.csv(results, "results/landscape_comparison_stats.csv", row.names=FALSE)

cat("\nStats saved to results/landscape_comparison_stats.rds\n")
