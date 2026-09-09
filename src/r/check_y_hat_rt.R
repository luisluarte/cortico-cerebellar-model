
source("eval_metrics_manual.R")
cat("Summary of y_hat_rt:\n")
print(summary(as.numeric(y_hat_rt)))
cat("Summary of data_list$y_rt:\n")
print(summary(as.numeric(data_list$y_rt)))
