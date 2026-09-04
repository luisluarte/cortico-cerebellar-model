library(ggplot2)

res <- readRDS('nsga2_crps_final.rds')
pareto_pars <- res$par
pareto_vals <- res$value

df <- data.frame(
  E = round(pareto_pars[, 1]),
  p = pareto_pars[, 2],
  CRPS = pareto_vals[, 1],
  Reliance = -pareto_vals[, 2] * 100
)

# Extract non-dominated points (mco automatically puts them in pareto.front() or the first few indices depending on nsga2 output)
# Wait, res$par is ALL evaluated points? No, nsga2() returns the final population, not all. 
# We can filter strictly for pareto dominance.
is_dominated <- function(i, df) {
  for (j in 1:nrow(df)) {
    if (i == j) next
    # Dominance: j dominates i if j is strictly better on at least one, and not worse on any
    # We want to MINIMIZE CRPS, MAXIMIZE Reliance
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

p_plot <- ggplot(df, aes(x = Reliance, y = CRPS, color = Pareto)) +
  geom_point(size = 3) +
  scale_color_manual(values = c("gray60", "red")) +
  geom_line(data = pareto_front[order(pareto_front$Reliance), ], aes(x = Reliance, y = CRPS), color="red", linetype="dashed") +
  theme_minimal() +
  labs(title = "NSGA-II Pareto Front (Gen 15)",
       subtitle = "Pathway Reliance vs. Absolute CRPS",
       x = "Pathway Reliance (%)",
       y = "Absolute CRPS (lower is better)")

ggsave("pareto_front.png", plot = p_plot, width = 7, height = 5, bg="white")
write.csv(pareto_front, "pareto_front.csv", row.names = FALSE)
