library(ggplot2)
library(dplyr)
library(mgcv)

fit_spline <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
long_df <- fit_spline$data

# Create the specific axis the user requested: (0 - Delta_Beta)
long_df$x_axis <- 0 - long_df$Delta_Beta

# Plotting the raw values (points) faceted by Condition
p <- ggplot(long_df, aes(x = x_axis, y = P_Calibrated, color = State)) +
  geom_point(alpha = 0.3, size = 1.5) +
  facet_wrap(~ Condition) +
  # Overlaying a stiff (k=4) GAM smooth so we can easily see the trend through the noise
  geom_smooth(method = "gam", formula = y ~ s(x, k=4), color = "black", linewidth = 1.2, se=FALSE) +
  scale_color_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  theme_bw(base_size = 14) +
  labs(
    title = "Raw Values: Calibrated P(Switch) across Task States",
    subtitle = "Points represent individual simulation probes. Black line = GAM trend (k=4).",
    x = bquote("( 0 - " ~ beta[thal] ~ " )\n<-- Fully Intact (-1.0)                     100% Ablated (0.0) -->"),
    y = "Calibrated Probability (0 to 1)"
  ) +
  theme(legend.position = "bottom")

ggsave("Fig43_Raw_Values_Facet.png", p, width = 10, height = 6, dpi=300)
cat("Done.\n")
