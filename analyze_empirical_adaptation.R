library(ggplot2)
library(dplyr)
library(zoo)

cat("Loading Empirical Targets and Posterior Parameters...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
lag_Reward <- targets$lag_reward
lag_Ch <- targets$lag_ch
t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")
beta_thal_subjs <- sapply(post_df[['bio_dist']][['traces']], function(x) exp(mean(x[, , 3])))

results <- data.frame()

for(s in 1:length(train_subjs)) {
  idx <- which(t_subjs == train_subjs[s])
  subj_reward <- lag_Reward[idx]
  subj_ch <- lag_Ch[idx]
  
  # Rolling Expected Value (10-trial moving average)
  EV <- rollmean(subj_reward, k=10, fill=NA, align="right")
  
  # Rolling Uncertainty (Entropy)
  eps_val <- 1e-5
  roll_win <- pmax(eps_val, pmin(1-eps_val, EV))
  unc <- -(roll_win * log2(roll_win) + (1-roll_win) * log2(1-roll_win))
  
  # Find Peaks in Uncertainty (Local Maxima where Uncertainty > 0.8)
  peaks <- c()
  for(t in 2:(length(unc)-5)) {
    if(!is.na(unc[t]) && !is.na(unc[t-1]) && !is.na(unc[t+1])) {
      if(unc[t] > unc[t-1] && unc[t] > unc[t+1] && unc[t] > 0.8) {
        peaks <- c(peaks, t)
      }
    }
  }
  
  if(length(peaks) == 0) next
  
  # Calculate Choice Switch Rate in the 5 trials following the peak
  switch_rates <- c()
  for(p in peaks) {
    if(p + 5 <= length(subj_ch)) {
      # Count how many times they switched choices in the next 5 trials
      switches <- sum(diff(subj_ch[p:(p+5)]) != 0, na.rm=TRUE)
      switch_rates <- c(switch_rates, switches / 5.0)
    }
  }
  
  mean_switch <- mean(switch_rates, na.rm=TRUE)
  
  results <- rbind(results, data.frame(
    Subject = s,
    Beta_Thal = beta_thal_subjs[s],
    Switch_Rate = mean_switch
  ))
}

# Compute Correlation
cor_res <- cor.test(results$Beta_Thal, results$Switch_Rate)
cat(sprintf("Pearson correlation: r = %.3f, p = %.4f\n", cor_res$estimate, cor_res$p.value))

# Plot
p <- ggplot(results, aes(x = Beta_Thal, y = Switch_Rate)) +
  geom_point(size = 4, color = "#d95f02", alpha = 0.8) +
  geom_smooth(method = "lm", color = "black", fill = "grey80", linewidth = 1.2) +
  labs(title = "Empirical Attractor Rigidity: \u03b2_thal vs Behavioral Flexibility",
       subtitle = sprintf("Higher Thalamic Feedback (\u03b2) locks behavior, reducing choice switching after uncertainty (r = %.2f, p = %.3f).", 
                          cor_res$estimate, cor_res$p.value),
       x = "Participant's Fitted Thalamic Feedback (\u03b2_thal)",
       y = "Empirical Choice Switch Rate Following High Uncertainty") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

ggsave("results/Fig8_Empirical_Beta_Adaptation.png", plot = p, width = 7, height = 5, dpi = 300)
cat("Saved to results/Fig8_Empirical_Beta_Adaptation.png\n")
