library(ggplot2)
library(dplyr)
library(tidyr)
library(broom)

cat("Loading Empirical Targets and Posterior Parameters...\n")
targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("final_model/results/factorial_fixed_10k_A.rds")

t_subjs <- targets$subjs
train_subjs <- unique(t_subjs)

# Extract ALL 7 subject-level generative parameters (exponentiated)
emp_params <- t(sapply(post_df[['bio_dist']][['traces']], function(x) apply(x, 3, mean)))
params_exp <- exp(emp_params)
colnames(params_exp) <- c("alpha_pc", "lambda_pc", "beta_thal", "kappa_cf", "alpha_gran", "beta_gran", "sigma2")

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
  
  # The User's Composite Metric: Markovian Behavior Index
  # 1.0 = Purely driven by t-1 (always Win-Stay, always Lose-Shift)
  # 0.5 = Random behavior independent of t-1
  markovian_index <- (win_stay + lose_shift) / 2.0
  
  res_row <- data.frame(
    Subject = s,
    Markovian_Index = markovian_index
  )
  res_row <- cbind(res_row, as.data.frame(t(params_exp[s, ])))
  results <- rbind(results, res_row)
}

# Standardize variables to get comparable Beta coefficients
results_scaled <- results %>%
  mutate(across(c(Markovian_Index, alpha_pc:sigma2), scale))

cat("Running Multiple Regression...\n")
fit <- lm(Markovian_Index ~ alpha_pc + lambda_pc + beta_thal + kappa_cf + alpha_gran + beta_gran + sigma2, data = results_scaled)
fit_summary <- summary(fit)
print(fit_summary)

# Extract tidy coefficients for Forest Plot
coef_df <- tidy(fit, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = recode(term, 
                  "alpha_pc" = "Cortical LR (\u03b1_pc)",
                  "lambda_pc" = "Cortical Decay (\u03bb_pc)",
                  "beta_thal" = "Thalamic Feedback (\u03b2_thal)",
                  "kappa_cf" = "Purkinje Plasticity (\u03ba_cf)",
                  "alpha_gran" = "Granule LR (\u03b1_gran)",
                  "beta_gran" = "Granule Decay (\u03b2_gran)",
                  "sigma2" = "Purkinje Drift (\u03c3\u00b2)"),
    # Sort by effect size
    term = reorder(term, estimate),
    significance = ifelse(p.value < 0.05, "Significant", "Non-Significant")
  )

cat("Generating Forest Plot of Generative Parameters...\n")

p <- ggplot(coef_df, aes(x = estimate, y = term, color = significance)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, linewidth = 1.2) +
  geom_point(size = 5) +
  scale_color_manual(values = c("Significant" = "#d95f02", "Non-Significant" = "grey50")) +
  labs(title = "Predicting the Markovian Behavior Index from the Generative Model",
       subtitle = "Multiple regression reveals how parameters interact to produce history-dependent vs Markovian behavior.",
       x = "Standardized Regression Coefficient (\u03b2)",
       y = "Generative Network Parameter") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.grid.minor = element_blank())

ggsave("results/Fig10_Markovian_Parameter_Regression.png", plot = p, width = 9, height = 6, dpi = 300)
cat("Done! Plot saved to results/Fig10_Markovian_Parameter_Regression.png\n")
