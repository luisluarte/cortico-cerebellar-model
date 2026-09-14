library(brms)
library(dplyr)
library(ggplot2)

fit <- readRDS("results/brms_global_parametric.rds")
df <- fit$data

ldb_seq <- seq(-2.3, 0.0, length.out = 100)
nd_opt <- data.frame(Condition = "Optimized", log_Delta_Beta = ldb_seq)
nd_les <- data.frame(Condition = "Lesioned", log_Delta_Beta = ldb_seq)

cat("Extracting posterior predictions on the RESPONSE scale...\n")
draws_opt <- fitted(fit, newdata = nd_opt, re_formula = NA, summary = FALSE)
draws_les <- fitted(fit, newdata = nd_les, re_formula = NA, summary = FALSE)
draws_diff <- draws_opt - draws_les

sum_diff <- data.frame(
  Delta_Beta = exp(ldb_seq),
  Diff_Median = apply(draws_diff, 2, median),
  Lower = apply(draws_diff, 2, quantile, probs = 0.025),
  Upper = apply(draws_diff, 2, quantile, probs = 0.975)
)

# Extract 1 data point per subject for biological density
df_subj <- df %>% group_by(Subject) %>% summarize(Delta_Beta = exp(first(log_Delta_Beta)))

# Compute empirical density
dens <- density(df_subj$Delta_Beta, from = min(sum_diff$Delta_Beta), to = max(sum_diff$Delta_Beta))
dens_df <- data.frame(x = dens$x, y = dens$y)

# Scale density to sit in the bottom 25% of the y-axis
y_min <- min(sum_diff$Lower) - 0.01
y_max <- max(sum_diff$Upper)
y_range <- y_max - y_min
dens_df$y_scaled <- y_min + (dens_df$y / max(dens_df$y)) * (y_range * 0.25)

# Calculate Median bound
median_val <- median(df_subj$Delta_Beta)

cat(sprintf("Truncating X-Axis between [%.3f, 0.00] based on [Median, 0]\n", median_val))

p <- ggplot(sum_diff, aes(x = Delta_Beta)) +
  geom_polygon(data = dens_df, aes(x = x, y = y_scaled), fill = "black", alpha = 0.15, inherit.aes = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "blue", alpha = 0.2) +
  geom_line(aes(y = Diff_Median), color = "blue", linewidth = 1.5) +
  scale_x_reverse() +
  coord_cartesian(ylim = c(y_min, y_max), xlim = c(median_val, 0.0)) +
  theme_bw(base_size = 15) +
  labs(
    title = "The Cerebellar Shock Absorber (Vulnerable Population)",
    subtitle = bquote(Delta ~ "P(Switch) [Optimized - Lesioned]. Axis truncated to [Median, 0]."),
    x = bquote("Structural Integrity (" ~ Delta*beta[thal] ~ ")  [Median -> Severe Damage]"),
    y = bquote("Change in P(Switch) [Cerebellar Correction]")
  )

ggsave("Fig65_Cerebellar_Correction_LowerHalf.png", p, width = 9, height = 6, dpi = 300)
cat("Done.\n")
