library(brms)
library(emmeans)
library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")
df <- fit_brm$data

# Create continuous grid spanning the full range of log_Delta_Beta
grid_log <- seq(min(df$log_Delta_Beta), max(df$log_Delta_Beta), length.out = 200)

# 1. Evaluate Absolute Probabilities
emm_abs <- emmeans(fit_brm, ~ Condition | log_Delta_Beta * State, 
                   at = list(log_Delta_Beta = grid_log), epred = TRUE)
sum_abs <- as.data.frame(emm_abs)

# EXPLICITLY TRANSFORM BACK TO LINEAR PERCENTAGE SCALE
sum_abs$Lesion_Percentage <- exp(sum_abs$log_Delta_Beta)

# 2. Evaluate the Paired Difference (The "Corrective Force")
emm_pairs <- pairs(emm_abs, by = c("log_Delta_Beta", "State"), reverse = TRUE)
sum_diff <- as.data.frame(emm_pairs)

# EXPLICITLY TRANSFORM BACK TO LINEAR PERCENTAGE SCALE
sum_diff$Lesion_Percentage <- exp(sum_diff$log_Delta_Beta)

# Panel A: The Absolute Crossing 
pA <- ggplot(sum_abs, aes(x = Lesion_Percentage, y = emmean, color = Condition, fill = Condition)) +
  facet_wrap(~ State) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1), breaks = seq(0, 1, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. The Behavioral Crossing Point (Linear Lesion Scale)",
    x = "",
    y = "Predicted P(Switch | Loss)"
  ) +
  theme(strip.text = element_text(face="bold"), legend.position="bottom")

# Panel B: The Corrective Force Vector
pB <- ggplot(sum_diff, aes(x = Lesion_Percentage, y = estimate)) +
  facet_wrap(~ State) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = lower.HPD, ymax = upper.HPD), alpha = 0.3, fill = "#984ea3") +
  geom_line(color = "#984ea3", linewidth = 1.5) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1), breaks = seq(0, 1, 0.25)) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Bidirectional Cerebellar Corrective Force",
    subtitle = "Δ P(Switch) = (Optimized - Lesioned)",
    x = "Ablation Magnitude (% of Cerebellar-Thalamic Pathway Severed)",
    y = "Cerebellar Force Vector\n(- = Stabilizes, + = Mobilizes)"
  ) +
  theme(strip.text = element_text(face="bold"))

# Combine
p_final <- pA / pB + plot_annotation(
  title = "The Cerebellum as a Bidirectional Homeostatic Controller",
  theme = theme(plot.title = element_text(size = 16, face = "bold"))
)

ggsave("Fig32_Bidirectional_Homeostasis_Linear.png", p_final, width = 10, height = 9, dpi = 300)
