library(ggplot2)
library(dplyr)
library(viridis)

# Load the historical swarm data
df <- readRDS("results/historical_swarm.rds")

# Transform Loss to Fitness (Higher is better)
df$Fitness <- 1 / df$Loss

# 1. Calculate the population centroid (mean) for each generation
centroids <- df %>%
  group_by(Generation) %>%
  summarize(
    Granule = mean(Granule),
    DCN = mean(DCN),
    Fitness = mean(Fitness)
  ) %>%
  arrange(Generation)

# 2. Fit a smoothed 2D surface to infer the Fitness Landscape (Peaks)
model <- loess(Fitness ~ Granule * DCN, data = df, span = 0.4)

# Create a dense grid to render the background gradient
grid_gc <- seq(min(df$Granule), max(df$Granule), length.out = 150)
grid_dcn <- seq(min(df$DCN), max(df$DCN), length.out = 150)
grid <- expand.grid(Granule = grid_gc, DCN = grid_dcn)

# Predict the smoothed fitness on the grid (coerce matrix to numeric vector)
grid$Fitness <- as.numeric(predict(model, newdata = grid))

# 3. Create the Visualization
p <- ggplot() +
  # Render the inferred smooth Fitness landscape (Gradient)
  geom_raster(data = grid, aes(x = Granule, y = DCN, fill = Fitness), interpolate = TRUE) +
  # Add subtle contour lines for topographical depth
  geom_contour(data = grid, aes(x = Granule, y = DCN, z = Fitness), color = "white", alpha = 0.2, bins = 15) +
  # Color palette (magma: bright = high fitness/peak, dark = low fitness/valley)
  scale_fill_viridis_c(option = "magma", name = "Fitness\n(1 / RMSE)") +
  
  # Overlay the faint scatter of the actual evaluations in the background
  geom_point(data = df, aes(x = Granule, y = DCN), color = "black", alpha = 0.15, size = 0.5) +
  
  # The Evolutionary Trajectory Path (Continuous line, no arrow)
  geom_path(data = centroids, aes(x = Granule, y = DCN), color = "cyan", linewidth = 1.2) +
  
  # Highlight the centroid stops (small dots along the path)
  geom_point(data = centroids, aes(x = Granule, y = DCN), color = "cyan", size = 2) +
  
  # The Red Cross at the final Global Maximum
  annotate("point", x = centroids$Granule[nrow(centroids)], y = centroids$DCN[nrow(centroids)], 
           color = "red", shape = 4, size = 6, stroke = 2.5) +
  
  # Annotate Start and End points
  annotate("text", x = centroids$Granule[1], y = centroids$DCN[1] - 7, 
           label = "Start (Gen 1)", color = "cyan", fontface = "bold", size = 5) +
  annotate("text", x = centroids$Granule[nrow(centroids)], y = centroids$DCN[nrow(centroids)] + 7, 
           label = "Global Maximum (Gen 20)", color = "red", fontface = "bold", size = 5) +
  
  # Labels and theming
  labs(
    title = "Evolutionary Trajectory Across the Fitness Landscape",
    subtitle = "Population centroid migrating toward the optimal structural bottleneck",
    x = "Granule Layer Size (N_G)",
    y = "DCN Layer Size (N_DCN)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right",
    panel.grid = element_blank() # Remove standard gridlines to emphasize the contour
  )

# Save the plots
ggsave("results/evolutionary_trajectory.png", plot = p, width = 10, height = 7, dpi = 300)
ggsave("results/evolutionary_trajectory.pdf", plot = p, width = 10, height = 7)
cat("Plot saved successfully to results/evolutionary_trajectory.png\n")
