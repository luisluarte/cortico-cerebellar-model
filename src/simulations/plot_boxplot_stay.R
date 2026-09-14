library(ggplot2)
library(dplyr)

fit <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
df <- fit$data

# Calculate P(Stay)
if("P_Switch" %in% colnames(df)) {
  df$P_Stay <- 1 - df$P_Switch
  ylab_text <- "Raw Probability of Staying"
} else {
  df$P_Stay <- 1 - df$P_Calibrated
  ylab_text <- "Calibrated Probability of Staying"
}

p <- ggplot(df, aes(x = Condition, y = P_Stay, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  geom_jitter(color = "black", alpha = 0.15, width = 0.15, size = 1.2) +
  facet_wrap(~ State) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Global Effect of Cerebellar Optimization on Perseveration",
    subtitle = "P(Stay | Loss) aggregated across all structural severities",
    y = ylab_text,
    x = "Network Architecture"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold")
  )

ggsave("Fig49_Boxplot_Stay.png", p, width = 8, height = 6, dpi = 300)
cat("Done.\n")
