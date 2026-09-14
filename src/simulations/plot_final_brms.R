library(ggplot2)
library(dplyr)
library(patchwork)

effects <- readRDS("results/final_brms_effects.rds")
ce_markovian <- effects$markovian
ce_non_markovian <- effects$non_markovian

# Panel A: Markovian Reactivity (Mean P_Switch)
p_mark <- ggplot(ce_markovian, aes(x = Delta_Beta, y = estimate__, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. Markovian Reactivity (1-Step)",
    subtitle = "Mean probability of switching immediately after a loss",
    x = bquote("Structural Integrity (" ~ beta[thal] ~ ")"),
    y = "P(Switch | Loss)"
  ) +
  theme(legend.position = "none")

# Panel B: Non-Markovian Perseveration (AR1)
p_non_mark <- ggplot(ce_non_markovian, aes(x = Delta_Beta, y = estimate__, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Non-Markovian Perseveration",
    subtitle = "Temporal sluggishness (AR1) of the probability trajectory",
    x = bquote("Structural Integrity (" ~ beta[thal] ~ ")"),
    y = "Autocorrelation (AR1)"
  ) +
  theme(legend.position = "bottom")

p_final <- p_mark + p_non_mark + plot_annotation(
  title = "Resolving the Paradox: Markovian Reactivity vs Non-Markovian Perseveration",
  subtitle = "Bayesian conditional effects across the continuous structural severity gradient",
  theme = theme(plot.title = element_text(size = 16, face = "bold"))
)

ggsave("Fig55_Final_Dual_Metrics.png", p_final, width = 12, height = 6, dpi = 300)
cat("Done.\n")
