library(dplyr)
df <- readRDS("results/reversal_df.rds")

cat("--- Stable State: P(Switch) Overall ---\n")
stable_overall <- df %>%
  filter(State == "Stable") %>%
  group_by(Condition) %>%
  summarize(Mean_P_Switch = mean(P_Switch), .groups="drop")
print(stable_overall)

cat("\n--- Stable State: P(Switch) by Feedback ---\n")
stable_fb <- df %>%
  filter(State == "Stable") %>%
  group_by(Condition, Feedback = ifelse(F==1, "Win", "Loss")) %>%
  summarize(Mean_P_Switch = mean(P_Switch), .groups="drop")
print(stable_fb)
