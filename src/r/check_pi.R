
source("eval_metrics_manual.R")
cat("Any NAs in Pi_mat? ", any(is.na(Pi_mat)), "\n")
cat("Summary of Pi_mat:\n")
print(summary(as.numeric(Pi_mat)))
