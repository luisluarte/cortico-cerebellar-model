library(ggplot2)

df <- read.csv('nsga2_phase2_checkpoint_results.csv', header = FALSE)
colnames(df) <- c("alpha", "beta", "D_compress", "CRPS", "Reliance")

# Remove any NA rows just in case
df <- df[complete.cases(df), ]

is_dominated <- function(i, df) {
  for (j in 1:nrow(df)) {
    if (i == j) next
    crps_better <- df$CRPS[j] <= df$CRPS[i]
    rel_better <- df$Reliance[j] >= df$Reliance[i]
    strict_crps <- df$CRPS[j] < df$CRPS[i]
    strict_rel <- df$Reliance[j] > df$Reliance[i]
    
    if (crps_better && rel_better && (strict_crps || strict_rel)) {
      return(TRUE)
    }
  }
  return(FALSE)
}

df$Pareto <- sapply(1:nrow(df), function(i) !is_dominated(i, df))
pareto_front <- df[df$Pareto, ]

# Scale Reliance to percentage
df$Reliance <- df$Reliance * 100
pareto_front$Reliance <- pareto_front$Reliance * 100

p_plot <- ggplot(df, aes(x = Reliance, y = CRPS, color = Pareto)) +
  geom_point(size = 3) +
  scale_color_manual(values = c("gray60", "blue")) +
  geom_line(data = pareto_front[order(pareto_front$Reliance), ], aes(x = Reliance, y = CRPS), color="blue", linetype="dashed") +
  theme_minimal() +
  labs(title = "Phase 2 NSGA-II Pareto Front (Gen 12/15)",
       subtitle = "Pathway Reliance vs. Absolute CRPS (Phantom Trace Local Plasticity)",
       x = "Pathway Reliance (%)",
       y = "Absolute CRPS (lower is better)")

ggsave("pareto_front_phase2.png", plot = p_plot, width = 7, height = 5, bg="white")
write.csv(pareto_front, "pareto_front_phase2.csv", row.names = FALSE)
