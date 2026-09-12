library(ggplot2)
library(dplyr)
library(viridis)

# Load the historical swarm data
df <- readRDS("results/historical_swarm.rds")

# Find the global optimum
best_idx <- which.min(df$Loss)
best_granule <- df$Granule[best_idx]
best_dcn <- df$DCN[best_idx]

# Create the Swarm Scatter Plot
p <- ggplot(df, aes(x = Granule, y = DCN)) +
  # Add the swarm of historical points, colored by Generation
  geom_point(aes(color = Generation, size = -Loss), alpha = 0.7) +
  # Highlight the optimum
  geom_point(data = data.frame(Granule=best_granule, DCN=best_dcn),
             color = "red", shape = 8, size = 5, stroke = 2) +
  # Color scale for the generations (shows time progression)
  scale_color_viridis_c(option = "magma", direction = -1) +
  # Labels and theme
  labs(
    title = "Evolutionary Swarm Tracking (Generational Drift)",
    subtitle = paste0("Global Optimum discovered at GC: ", best_granule, ", DCN: ", best_dcn),
    x = "Granule Layer Size (N_G)",
    y = "DCN Layer Size (N_DCN)",
    color = "Generation\n(Time)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  ) +
  # Do not show legend for the size mapping
  guides(size = "none")

# Save the plot
ggsave("results/historical_swarm_plot.png", plot = p, width = 9, height = 7, dpi = 300)
ggsave("results/historical_swarm_plot.pdf", plot = p, width = 9, height = 7)
cat("Plot saved successfully to results/historical_swarm_plot.png\n")
