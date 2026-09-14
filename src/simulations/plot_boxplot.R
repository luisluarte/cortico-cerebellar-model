library(ggplot2)
library(dplyr)

fit_spline <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
long_df <- fit_spline$data

p <- ggplot(long_df, aes(x = Condition, y = P_Calibrated, fill = Condition)) +
  # Add the boxplot, hiding the default outliers since geom_jitter will plot all points anyway
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  # Overlay the raw data points with a slight jitter so they don't perfectly overlap
  geom_jitter(color = "black", alpha = 0.15, width = 0.15, size = 1.2) +
  facet_wrap(~ State) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Global Effect of Cerebellar Optimization",
    subtitle = "Aggregated across all lesion severities",
    y = "Calibrated Probability of Switching",
    x = "Network Architecture"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold")
  )

ggsave("Fig44_Boxplot_Condition.png", p, width = 8, height = 6, dpi = 300)
cat("Done.\n")
