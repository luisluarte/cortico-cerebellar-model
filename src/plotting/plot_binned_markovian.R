library(ggplot2)
library(dplyr)

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
  
  results <- rbind(results, data.frame(
    Subject = s, 
    MI = mi, 
    Beta_Thal = beta_thal_subjs[s]
  ))
}

cat("Generating Binned Plot...\n")

p <- ggplot(results, aes(x = MI, y = Beta_Thal)) +
  # Raw data points in the background
  geom_jitter(color = "grey60", alpha = 0.6, width = 0.005, size = 2.5) +
  # Binned averages (using stat_summary_bin for robust internal binning)
  stat_summary_bin(bins = 6, fun.data = "mean_se", geom = "errorbar", width = 0.015, color = "#d95f02", linewidth = 1.2) +
  stat_summary_bin(bins = 6, fun = "mean", geom = "line", color = "#d95f02", linewidth = 1.2) +
  stat_summary_bin(bins = 6, fun = "mean", geom = "point", size = 5, color = "#d95f02", fill = "white", shape = 21, stroke = 1.5) +
  labs(title = "Non-Linear U-Shape: Thalamic Feedback vs Heuristic Behavior",
       subtitle = "Binned \u03b2_thal reveals that both highly rigid (non-Markovian) and highly reactive (Markovian)\nstrategies are driven by an over-amplified Thalamic Attractor.",
       x = "Empirical Markovian Behavior Index (MI Bins)",
       y = "Mean Thalamic Feedback (\u03b2_thal)") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave("results/Fig13_Binned_Markovian_Beta.png", plot = p, width = 8, height = 6, dpi = 300)
cat("Done! Plot saved to results/Fig13_Binned_Markovian_Beta.png\n")
