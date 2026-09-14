library(brms)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")
df <- fit_brm$data

# Create a dense continuous grid spanning the full range of actual lesion magnitudes
grid_log <- seq(min(df$log_Delta_Beta), max(df$log_Delta_Beta), length.out = 100)

# 1. Evaluate Absolute Probabilities across the continuous continuum
emm_abs <- emmeans(fit_brm, ~ Condition | log_Delta_Beta * State, 
                   at = list(log_Delta_Beta = grid_log), epred = TRUE)
sum_abs <- as.data.frame(emm_abs)

# 2. Evaluate the Paired Difference (The "Corrective Force")
# 'reverse = TRUE' ensures the contrast is Optimized - Lesioned
emm_pairs <- pairs(emm_abs, by = c("log_Delta_Beta", "State"), reverse = TRUE)
sum_diff <- as.data.frame(emm_pairs)

# Panel A: The Absolute Crossing 
pA <- ggplot(sum_abs, aes(x = log_Delta_Beta, y = emmean, color = Condition, fill = Condition)) +
  facet_wrap(~ State) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. The Behavioral Crossing Point",
    subtitle = "The Optimized intact network structurally crosses the Lesioned behavior",
    x = "",
    y = "Predicted P(Switch | Loss)"
  ) +
  theme(strip.text = element_text(face="bold"), legend.position="bottom")

# Panel B: The Corrective Force Vector
pB <- ggplot(sum_diff, aes(x = log_Delta_Beta, y = estimate)) +
  facet_wrap(~ State) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.3, fill = "#984ea3") +
  geom_line(color = "#984ea3", linewidth = 1.5) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Bidirectional Cerebellar Corrective Force",
    subtitle = "Δ P(Switch) = (Optimized - Lesioned)",
    x = "Lesion Magnitude: log(Δ β thalamus)\n<-- Small Lesions (Hyper-volatile)         Large Lesions (Stuck/Perseverative) -->",
    y = "Cerebellar Force Vector\n(- = Stabilizes, + = Mobilizes)"
  ) +
  theme(strip.text = element_text(face="bold"))

# Combine
p_final <- pA / pB + plot_annotation(
  title = "The Cerebellum as a Bidirectional Homeostatic Controller",
  theme = theme(plot.title = element_text(size = 16, face = "bold"))
)

ggsave("Fig31_Bidirectional_Homeostasis.png", p_final, width = 10, height = 9, dpi = 300)
