
source("eval_metrics_manual.R")
cat("Mean of true_labels: ", mean(true_labels), "\n")
cat("Any NAs in y_hat_choice? ", any(is.na(y_hat_choice)), "\n")
cat("Summary of y_hat_choice:\n")
print(summary(y_hat_choice))
