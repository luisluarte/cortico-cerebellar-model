library(dplyr)
library(ggplot2)
library(gridExtra)

df <- readRDS("results/reversal_df.rds")

# Compute the variability (Standard Deviation) of P_Switch across all trials for each subject
var_df <- df %>%
  group_by(Subject, Condition, log_Delta_Beta) %>%
  summarize(
    P_Switch_SD = sd(P_Switch),
    P_Switch_Mean = mean(P_Switch),
    .groups = "drop"
  )

var_df$Delta_Beta <- exp(var_df$log_Delta_Beta)

cat("Summary of P_Switch Variability (SD):\n")
var_summary <- var_df %>% 
  group_by(Condition) %>% 
  summarize(Mean_SD = mean(P_Switch_SD), Median_SD = median(P_Switch_SD))
print(var_summary)

# Panel 1: Overall Boxplot of Variability
p_box <- ggplot(var_df, aes(x = Condition, y = P_Switch_SD, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "A. Dynamic Policy Adjustment",
    subtitle = "Variability (SD) of P(Switch) across all trials",
    y = "Trial-to-Trial SD of P(Switch)",
    x = ""
  ) +
  theme(legend.position = "none")

# Panel 2: Variability as a function of Structural Damage
p_scatter <- ggplot(var_df, aes(x = Delta_Beta, y = P_Switch_SD, color = Condition)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.5) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_reverse() +
  theme_bw(base_size = 15) +
  labs(
    title = "B. Impact of Structural Damage",
    subtitle = "Cerebellar ablation destroys policy variance",
    x = bquote("Structural Integrity (" ~ Delta*beta[thal] ~ ") [Health -> Damage]"),
    y = "Trial-to-Trial SD of P(Switch)"
  ) +
  theme(legend.position = "bottom")

p_final <- grid.arrange(p_box, p_scatter, ncol = 2, top = "Loss of Dynamic Context Integration")
ggsave("Fig67_Policy_Variability.png", p_final, width = 12, height = 6, dpi = 300)
cat("Done.\n")
