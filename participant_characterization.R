library(ggplot2)
library(dplyr)
library(tidyr)

cat("Loading Empirical Targets and Posterior Parameters...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

# Extract subject-level beta_thal (exponentiated from unconstrained posterior trace)
beta_thal_subjs <- sapply(post_df[['bio_dist']][['traces']], function(x) exp(mean(x[, , 3])))

results <- data.frame()

for(s in 1:length(train_subjs)) {
  idx <- which(t_subjs == train_subjs[s])
  
  # Extract behavioral vectors for this subject
  ch <- targets$lag_ch[idx]
  prev_rew <- targets$lag_reward[idx]
  resp_time <- targets$lag_resp[idx] 
  
  # Compute boolean masks using shifted choices
  n_trials <- length(ch)
  cur_ch <- ch[2:n_trials]
  prev_ch <- ch[1:(n_trials-1)]
  prev_rew <- prev_rew[1:(n_trials-1)]
  
  if(!is.null(resp_time)) {
      resp_time <- resp_time[2:n_trials]
  } else {
      resp_time <- rep(NA, n_trials - 1)
  }
  
  # Clean NAs
  valid <- !is.na(cur_ch) & !is.na(prev_ch) & !is.na(prev_rew)
  cur_ch <- cur_ch[valid]
  prev_ch <- prev_ch[valid]
  prev_rew <- prev_rew[valid]
  resp_time <- resp_time[valid]
  
  # Compute boolean masks
  switch_trial <- (cur_ch != prev_ch)
  stay_trial <- (cur_ch == prev_ch)
  
  win_idx <- (prev_rew == 1)
  loss_idx <- (prev_rew == 0)
  
  # Compute metrics
  mean_reward <- mean(prev_rew)
  switch_rate <- mean(switch_trial)
  win_stay <- ifelse(sum(win_idx) > 0, mean(stay_trial[win_idx]), NA)
  lose_shift <- ifelse(sum(loss_idx) > 0, mean(switch_trial[loss_idx]), NA)
  lose_stay <- ifelse(sum(loss_idx) > 0, mean(stay_trial[loss_idx]), NA) # Persistence
  mean_rt <- mean(resp_time, na.rm=TRUE)
  
  results <- rbind(results, data.frame(
    Subject = s,
    Beta_Thal = beta_thal_subjs[s],
    `1. Mean Reward` = mean_reward,
    `2. Overall Switch Rate` = switch_rate,
    `3. Win-Stay (Adherence)` = win_stay,
    `4. Lose-Shift (Flexibility)` = lose_shift,
    `5. Lose-Stay (Persistence)` = lose_stay,
    `6. Mean Response Time` = mean_rt,
    check.names = FALSE
  ))
}

# Convert to long format for faceting
df_long <- results %>%
  pivot_longer(cols = `1. Mean Reward`:`6. Mean Response Time`, 
               names_to = "Metric", values_to = "Value") %>%
  filter(!is.na(Value)) # Drop metrics that couldn't be computed (e.g. missing RT)

# Compute correlations per metric for annotations
cor_labels <- df_long %>%
  group_by(Metric) %>%
  summarize(
    r = cor(Beta_Thal, Value, use="complete.obs"),
    p = cor.test(Beta_Thal, Value)$p.value,
    x_pos = max(df_long$Beta_Thal, na.rm=TRUE) * 0.9,
    y_pos = max(Value, na.rm=TRUE)
  ) %>%
  mutate(
    # Highlight significant correlations with an asterisk
    sig = ifelse(p < 0.05, "*", ""),
    label = sprintf("r = %.2f%s\np = %.3f", r, sig, p)
  )

cat("Generating Behavioral Characterization Plot...\n")

p <- ggplot(df_long, aes(x = Beta_Thal, y = Value)) +
  geom_point(color = "#2a9d7a", size = 2.5, alpha = 0.7) +
  geom_smooth(method = "lm", color = "black", fill = "grey80", linewidth = 1) +
  facet_wrap(~Metric, scales = "free_y", ncol = 3) +
  geom_text(data = cor_labels, aes(x = x_pos, y = y_pos, label = label), 
            vjust = 1, hjust = 1, size = 4, fontface = "bold", color = "black") +
  labs(title = "Full Behavioral Characterization vs Thalamic Feedback (\u03b2_thal)",
       subtitle = "Translating the fitted generative attractor rigidity into overt human behavioral strategies (N=50).",
       x = "Participant's Empirical Thalamic Feedback (\u03b2_thal)",
       y = "Behavioral Metric Value") +
  theme_minimal(base_size = 14) +
  theme(strip.background = element_rect(fill = "#e0e0e0", color = "black"),
        strip.text = element_text(face = "bold", size = 11),
        panel.border = element_rect(color = "black", fill = NA),
        panel.grid.minor = element_blank())

ggsave("results/Fig9_Beta_Behavior_Characterization.png", plot = p, width = 12, height = 8, dpi = 300)
cat("Done! Plot saved to results/Fig9_Beta_Behavior_Characterization.png\n")
