library(dplyr)

fit <- readRDS("results/brms_switch_model_calibrated.rds")
df <- fit$data

cat("--- Summary of Model Data ---\n")
print(table(df$State, df$Condition))

cat("\n--- Mean P_Calibrated by State and Condition ---\n")
df %>%
  group_by(State, Condition) %>%
  summarize(
    N = n(),
    Mean_P = mean(P_Calibrated),
    SD_P = sd(P_Calibrated),
    .groups = "drop"
  ) %>%
  print()

cat("\n--- Distribution of log_Delta_Beta ---\n")
print(summary(df$log_Delta_Beta))
