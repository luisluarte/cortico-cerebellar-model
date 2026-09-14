library(dplyr)
df <- readRDS("results/reversal_df.rds")

cat("--- Policy Breakdown (P_Switch) ---\n")
policy <- df %>%
  group_by(Condition, Feedback = ifelse(F==1, "Win", "Loss")) %>%
  summarize(
    Mean_P_Switch = mean(P_Switch),
    .groups = "drop"
  )
print(policy)
