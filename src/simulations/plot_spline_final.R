library(brms)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)

fit_spline <- readRDS("results/brms_switch_model_spline_global.rds")

# We evaluate the grid DIRECTLY on the raw Delta_Beta scalar, avoiding all log artifacts
grid_beta <- seq(0.01, 1.0, length.out = 150) 

# 1. Evaluate Absolute Probabilities across the spline
emm_abs <- emmeans(fit_spline, ~ Condition | Delta_Beta * State, 
                   at = list(Delta_Beta = grid_beta), epred = TRUE)
sum_abs <- as.data.frame(emm_abs)

# 2. Evaluate the Paired Difference (The non-linear "Corrective Force")
emm_pairs <- pairs(emm_abs, by = c("Delta_Beta", "State"), reverse = TRUE)
sum_diff <- as.data.frame(emm_pairs)

# Panel A: The Absolute Spline Curves
pA <- ggplot(sum_abs, aes(x = Delta_Beta, y = emmean, color = Condition, fill = Condition)) +
  facet_wrap(~ State) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. Non-Linear Behavioral Curves (GAM Splines)",
    x = "",
    y = "Predicted P(Switch | Loss)"
  ) +
  theme(strip.text = element_text(face="bold"), legend.position="bottom")

# Panel B: The Empirical Corrective Force Vector
pB <- ggplot(sum_diff, aes(x = Delta_Beta, y = estimate)) +
  facet_wrap(~ State) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.3, fill = "#984ea3") +
  geom_line(color = "#984ea3", linewidth = 1.5) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Empirically Fitted Cerebellar Corrective Force",
    subtitle = "Δ P(Switch) = (Optimized - Lesioned)",
    x = bquote("Thalamic Connection Scalar (" ~ beta[thal] ~ ")\n<-- 100% Ablation (Disconnected)                     Fully Intact (1.0) -->"),
    y = "Cerebellar Force Vector\n(- = Stabilizes, + = Mobilizes)"
  ) +
  theme(strip.text = element_text(face="bold"))

# Combine
p_final <- pA / pB + plot_annotation(
  title = "Non-Linear Spline Model of Bidirectional Homeostasis",
  subtitle = "Generalized Additive Model fitted to raw ablation scalars. Zero mathematical extrapolations.",
  theme = theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, face = "italic", color="grey30")
  )
)

ggsave("Fig35_Spline_Homeostasis_Final.png", p_final, width = 10, height = 9, dpi = 300)
