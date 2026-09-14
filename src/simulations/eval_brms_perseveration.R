library(brms)
library(ggplot2)
library(dplyr)

fit_brm <- readRDS("results/brms_perseveration_interaction.rds")
ce <- readRDS("results/brms_perseveration_ce.rds")

cat("--- Bayesian Beta Regression (Severe Group) ---\n")
print(summary(fit_brm))

# Plot the Conditional Effects (Interaction)
p_data <- ce[[1]]

p <- ggplot(p_data, aes(x = Loss_Streak_Scale, y = estimate__, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Bayesian Posterior: Perseveration Interaction",
    subtitle = "P(Stay | N consecutive losses) in severely damaged networks (Beta Regression)",
    x = "Consecutive Losses (Scaled)",
    y = "Posterior P(Stay)"
  ) +
  theme(legend.position = "bottom")

ggsave("Fig54_Bayesian_Perseveration.png", p, width = 8, height = 6, dpi = 300)
cat("Done.\n")
