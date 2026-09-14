library(ggplot2)
library(dplyr)
library(patchwork)

ce <- readRDS("results/brms_reversal_gam_ce.rds")

# ce[[1]] contains the marginal effects
plot_data <- ce[[1]]

# The Cond_State factor is "Condition.State"
# Let's split it back into Condition and State for clean plotting
plot_data$Condition <- sapply(strsplit(as.character(plot_data$Cond_State), "\\."), `[`, 1)
plot_data$State <- sapply(strsplit(as.character(plot_data$Cond_State), "\\."), `[`, 2)

# Ensure proper ordering
plot_data$Condition <- factor(plot_data$Condition, levels = c("Lesioned", "Optimized"))
plot_data$State <- factor(plot_data$State, levels = c("Stable", "Reversal"))

p <- ggplot(plot_data, aes(x = log_Delta_Beta, y = estimate__, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__), alpha = 0.2, color = NA) +
  facet_wrap(~ State) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Cerebellar Buffering: Independent State Dynamics",
    subtitle = "Bayesian GAM Splines fitted independently for Stable vs Reversal states",
    x = bquote("Structural Integrity (" ~ log(Delta * beta) ~ ")"),
    y = "Calibrated P(Switch)"
  ) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

ggsave("Fig56_Reversal_Specific_GAM.png", p, width = 10, height = 6, dpi = 300)
cat("Done.\n")
