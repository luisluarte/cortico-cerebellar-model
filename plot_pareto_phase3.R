library(ggplot2)

df <- read.csv('nsga2_phase3_checkpoint_results.csv', header = FALSE)
colnames(df) <- c("alpha", "beta", "D_compress", "sigma_sq", "CRPS", "Reliance")

# Remove any NA rows
df <- df[complete.cases(df), ]
df <- df[df$CRPS < 5, ] # filter out the ones that failed (CRPS = 10)

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

if (nrow(df) > 0) {
  df$Pareto <- sapply(1:nrow(df), function(i) !is_dominated(i, df))
  pareto_front <- df[df$Pareto, ]

  # Scale Reliance to percentage
  df$Reliance <- df$Reliance * 100
  pareto_front$Reliance <- pareto_front$Reliance * 100

  p_plot <- ggplot(df, aes(x = Reliance, y = CRPS, color = Pareto)) +
    geom_point(size = 3) +
    scale_color_manual(values = c("gray60", "purple")) +
    geom_line(data = pareto_front[order(pareto_front$Reliance), ], aes(x = Reliance, y = CRPS), color="purple", linetype="dashed") +
    theme_minimal() +
    labs(title = "Phase 3 NSGA-II Pareto Front (Gen 12)",
         subtitle = "Pathway Reliance vs. Absolute CRPS (Thermodynamic Diffusion)",
         x = "Pathway Reliance (%)",
         y = "Absolute CRPS (lower is better)")

  ggsave("pareto_front_phase3.png", plot = p_plot, width = 7, height = 5, bg="white")
  write.csv(pareto_front, "pareto_front_phase3.csv", row.names = FALSE)
} else {
  cat("No valid points found\\n")
}
