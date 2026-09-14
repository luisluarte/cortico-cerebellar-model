library(brms)
library(dplyr)
library(ggplot2)

fit <- readRDS("results/brms_reversal_gam.rds")

# Generate sequence of log_Delta_Beta covering the empirical range
ldb_seq <- seq(-2.3, 0.0, length.out = 50)

# Create newdata for Reversal state only
nd_opt <- data.frame(
  Condition = "Optimized",
  State = "Reversal",
  log_Delta_Beta = ldb_seq,
  Cond_State = interaction("Optimized", "Reversal")
)
nd_les <- data.frame(
  Condition = "Lesioned",
  State = "Reversal",
  log_Delta_Beta = ldb_seq,
  Cond_State = interaction("Lesioned", "Reversal")
)

# Get posterior epreds (population-level effects, ignoring Subject RE)
cat("Extracting posterior draws...\n")
draws_opt <- fitted(fit, newdata = nd_opt, re_formula = NA, summary = FALSE)
draws_les <- fitted(fit, newdata = nd_les, re_formula = NA, summary = FALSE)

# Calculate difference (Optimized - Lesioned) for each draw at each Delta_Beta
diff_draws <- draws_opt - draws_les

# Summarize the contrasts
summary_df <- data.frame(
  log_Delta_Beta = ldb_seq,
  Median_Diff = apply(diff_draws, 2, median),
  Lower_95 = apply(diff_draws, 2, quantile, probs = 0.025),
  Upper_95 = apply(diff_draws, 2, quantile, probs = 0.975),
  Prob_Direction = apply(diff_draws, 2, function(x) mean(x > 0))
)

cat("\n--- Statistical Contrast: Optimized - Lesioned (Reversal State) ---\n")
print(summary_df %>% 
      filter(row_number() %% 10 == 1) %>% # print subset for readability
      mutate(across(where(is.numeric), round, 4)))

# Plot the statistical difference
p_diff <- ggplot(summary_df, aes(x = log_Delta_Beta, y = Median_Diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  geom_ribbon(aes(ymin = Lower_95, ymax = Upper_95), alpha = 0.3, fill = "blue") +
  geom_line(color = "blue", linewidth = 1.2) +
  theme_bw(base_size = 14) +
  labs(
    title = "Bayesian Contrast in Reversal: Optimized - Lesioned",
    subtitle = "95% Credible Interval of the difference. If ribbon crosses 0, difference is not strictly significant.",
    x = bquote("Structural Integrity (" ~ log(Delta * beta) ~ ")"),
    y = expression(Delta ~ "P(Switch) [Opt - Les]")
  )

ggsave("Fig57_Reversal_Contrast.png", p_diff, width = 8, height = 6, dpi = 300)
cat("Done.\n")
