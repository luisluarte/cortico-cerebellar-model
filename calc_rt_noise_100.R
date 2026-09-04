library(jsonlite)
library(dplyr)
stan_data <- read_json('data/stan_data_N100.json', simplifyVector = TRUE)

df <- data.frame(subj = stan_data$subj, RT = stan_data$RT)

pooled_var_calc <- df %>%
  group_by(subj) %>%
  summarize(
    var_rt = var(RT),
    n_trials = n()
  ) %>%
  summarize(
    pooled_var = sum((n_trials - 1) * var_rt) / sum(n_trials - 1)
  )

pooled_sd <- sqrt(pooled_var_calc$pooled_var)

cat(sprintf("\n=== RT Noise Baseline (N=100) ===\n"))
cat(sprintf("Pooled Intra-Individual Variance: %.4f\n", pooled_var_calc$pooled_var))
cat(sprintf("Square Root (Pooled SD): %.4f\n", pooled_sd))
cat(sprintf("=================================\n"))
