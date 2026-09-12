library(ggplot2)
library(dplyr)

cat("Loading Empirical Data...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

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
  
  results <- rbind(results, data.frame(Subject = s, MI = mi))
}

cat("Generating Distribution Plot...\n")
mean_mi <- mean(results$MI, na.rm=TRUE)
max_density <- max(density(results$MI, na.rm=TRUE)$y)

p <- ggplot(results, aes(x = MI)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15, fill = "#7572b8", color = "black", alpha = 0.7) +
  geom_density(color = "black", linewidth = 1.2) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "#d95f02", linewidth = 1.2) +
  geom_vline(xintercept = mean_mi, linetype = "solid", color = "#1f78b4", linewidth = 1.2) +
  annotate("text", x = 0.5, y = max_density * 1.05, label = "Random Baseline (0.5)", color = "#d95f02", hjust = -0.05, fontface = "bold") +
  annotate("text", x = mean_mi, y = max_density * 1.05, label = sprintf("Mean (%.2f)", mean_mi), color = "#1f78b4", hjust = 1.1, fontface = "bold") +
  labs(title = "Empirical Distribution of the Markovian Behavior Index (MI)",
       subtitle = "Measures adherence to purely t-1 dependent behavior (Win-Stay/Lose-Shift) across 50 humans.",
       x = "Markovian Behavior Index (MI)",
       y = "Density") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

ggsave("results/Fig11_Markovian_Distribution.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Done! Plot saved to results/Fig11_Markovian_Distribution.png\n")
