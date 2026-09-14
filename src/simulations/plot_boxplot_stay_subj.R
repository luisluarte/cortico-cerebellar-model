library(ggplot2)
library(dplyr)

fit <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
df <- fit$data

# Calculate P(Stay)
if("P_Switch" %in% colnames(df)) {
  df$P_Stay <- 1 - df$P_Switch
  ylab_text <- "Mean Raw Probability of Staying"
} else {
  df$P_Stay <- 1 - df$P_Calibrated
  ylab_text <- "Mean Calibrated Probability of Staying"
}

# Average per participant!
df_subj <- df %>%
  group_by(Subject, Condition, State) %>%
  summarize(P_Stay = mean(P_Stay, na.rm=TRUE), .groups = "drop")

p <- ggplot(df_subj, aes(x = Condition, y = P_Stay, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  # Points are now 1 per subject per condition, making them much clearer
  geom_jitter(color = "black", alpha = 0.5, width = 0.15, size = 1.5) +
  facet_wrap(~ State) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Global Effect of Cerebellar Optimization on Perseveration",
    subtitle = "Points represent the mean P(Stay | Loss) per simulated participant",
    y = ylab_text,
    x = "Network Architecture"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold")
  )

ggsave("Fig50_Boxplot_Stay_Subject.png", p, width = 8, height = 6, dpi = 300)
cat("Done.\n")
