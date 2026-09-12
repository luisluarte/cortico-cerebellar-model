library(ggplot2)
library(dplyr)
library(mgcv)

cat("Loading Empirical Data...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)
beta_thal_subjs <- sapply(post_df[['bio_dist']][['traces']], function(x) exp(mean(x[, , 3])))

results <- data.frame()

for(s in 1:length(train_subjs)) {
  idx <- which(t_subjs == train_subjs[s])
  
  ch <- targets$lag_ch[idx]
  prev_rew <- targets$lag_reward[idx]
  
  n_trials <- length(ch)
  cur_ch <- ch[2:n_trials]
  prev_ch <- ch[1:(n_trials-1)]
  prev_rew <- prev_rew[1:(n_trials-1)]
  
  valid <- !is.na(cur_ch) & !is.na(prev_ch) & !is.na(prev_rew)
  cur_ch <- cur_ch[valid]
  prev_ch <- prev_ch[valid]
  prev_rew <- prev_rew[valid]
  
  switch_trial <- (cur_ch != prev_ch)
  stay_trial <- (cur_ch == prev_ch)
  win_idx <- (prev_rew == 1)
  loss_idx <- (prev_rew == 0)
  
  win_stay <- ifelse(sum(win_idx) > 0, mean(stay_trial[win_idx]), NA)
  lose_shift <- ifelse(sum(loss_idx) > 0, mean(switch_trial[loss_idx]), NA)
  
  mi <- (win_stay + lose_shift) / 2.0
  mean_reward <- mean(prev_rew)
  
  results <- rbind(results, data.frame(
    Subject = s, 
    MI = mi, 
    Mean_Reward = mean_reward,
    Beta_Thal = beta_thal_subjs[s]
  ))
}

# Axis 2: Relative Performance (Z-score compared to the sample)
results$Relative_Performance <- scale(results$Mean_Reward)[,1]

cat("Fitting 2D Surface GAM...\n")
# Fit a continuous thin-plate spline to interpolate beta_thal across the 2D behavioral space
model <- gam(Beta_Thal ~ s(MI, Relative_Performance, k=15), data = results)

# Create a dense grid for geom_raster
mi_seq <- seq(min(results$MI), max(results$MI), length.out = 200)
rp_seq <- seq(min(results$Relative_Performance), max(results$Relative_Performance), length.out = 200)
grid <- expand.grid(MI = mi_seq, Relative_Performance = rp_seq)

# Predict the beta_thal values for the landscape
grid$Beta_Thal_Pred <- predict(model, grid)

# Ensure the raster doesn't extrapolate too wildly outside the convex hull of the data
# (Clamp extreme predictions just for visual consistency of the heatmap scale)
p95 <- quantile(results$Beta_Thal, 0.95)
grid$Beta_Thal_Pred <- pmax(min(results$Beta_Thal), pmin(grid$Beta_Thal_Pred, p95))

cat("Plotting 3-Axis Behavioral Landscape...\n")

p <- ggplot() +
  # Continuous smooth raster background
  geom_raster(data = grid, aes(x = MI, y = Relative_Performance, fill = Beta_Thal_Pred), alpha = 0.8) +
  # White topographical contour lines
  geom_contour(data = grid, aes(x = MI, y = Relative_Performance, z = Beta_Thal_Pred), color = "white", alpha = 0.5, bins = 10) +
  # Actual empirical participant points
  geom_point(data = results, aes(x = MI, y = Relative_Performance, fill = Beta_Thal), 
             shape = 21, size = 4, color = "black", stroke = 1.2) +
  scale_fill_viridis_c(name = "Thalamic\nFeedback\n(\u03b2_thal)", option = "magma") +
  labs(title = "Behavioral Topography of Cortico-Cerebellar Rigidity",
       subtitle = "Interpolated 2D surface mapping how \u03b2_thal drives the interaction between Strategy (MI) and Performance.",
       x = "Markovian Behavior Index (Heuristic Strategy)",
       y = "Relative Performance (Z-Scored Reward)") +
  theme_minimal(base_size = 14) +
  coord_cartesian(expand = FALSE) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(face = "bold"),
        legend.position = "right")

ggsave("results/Fig12_Behavioral_Beta_Landscape.png", plot = p, width = 10, height = 7, dpi = 300)
cat("Done! Plot saved to results/Fig12_Behavioral_Beta_Landscape.png\n")
