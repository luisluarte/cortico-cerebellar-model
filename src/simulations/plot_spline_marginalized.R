library(brms)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)

fit_spline <- readRDS("results/brms_switch_model_calibrated_simplified.rds")

grid_beta <- seq(0.01, 1.0, length.out = 150) 

# 1. Evaluate Absolute Probabilities, MARGINALIZING over State
emm_abs <- emmeans(fit_spline, ~ Condition | Delta_Beta, 
                   at = list(Delta_Beta = grid_beta), epred = TRUE)
sum_abs <- as.data.frame(emm_abs)

# 2. Evaluate the Paired Difference
emm_pairs <- pairs(emm_abs, by = "Delta_Beta", reverse = TRUE)
sum_diff <- as.data.frame(emm_pairs)

# Panel A (No facets)
pA <- ggplot(sum_abs, aes(x = Delta_Beta, y = emmean, color = Condition, fill = Condition)) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. Universal Non-Linear Behavioral Curves",
    x = "",
    y = "Predicted P(Switch | Loss)"
  ) +
  theme(legend.position="bottom")

# Panel B (No facets)
pB <- ggplot(sum_diff, aes(x = Delta_Beta, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.3, fill = "#984ea3") +
  geom_line(color = "#984ea3", linewidth = 1.5) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Universal Cerebellar Corrective Force",
    subtitle = "Δ P(Switch) = (Optimized - Lesioned)",
    x = bquote("Thalamic Connection Scalar (" ~ beta[thal] ~ ")\n<-- 100% Ablation (Disconnected)                     Fully Intact (1.0) -->"),
    y = "Cerebellar Force Vector\n(- = Stabilizes, + = Mobilizes)"
  )

# Combine
p_final <- pA / pB + plot_annotation(
  title = "Universal Non-Linear Spline Model of Bidirectional Homeostasis",
  subtitle = "Marginalized across task states. Divergences: 0. Rhat: 1.00.",
  theme = theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, face = "italic", color="grey30")
  )
)

ggsave("Fig37_Spline_Marginalized.png", p_final, width = 8, height = 9, dpi = 300)
