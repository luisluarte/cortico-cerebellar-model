library(brms)
library(dplyr)

long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)

# NO MIN-MAX SCALING. 
# We only apply the mathematical boundary squeeze required by the Beta distribution.
long_df$P_Raw_Squeezed <- pmax(pmin(long_df$P_Switch, 0.9999), 0.0001)

raw_spline_formula <- bf(
  P_Raw_Squeezed ~ Condition * State + s(Delta_Beta, by = Condition) + (1 | Subject),
  phi ~ 1
)

print("Fitting Non-Linear Spline Model on RAW Probabilities...")

fit_raw <- brm(
  formula = raw_spline_formula,
  data = long_df,
  family = Beta(link = "logit", link_phi = "log"),
  chains = 4, cores = 4, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 12),
  seed = 42
)

saveRDS(fit_raw, "results/brms_switch_model_raw_spline.rds")

sink("results/brms_switch_summary_raw_spline.txt")
print(summary(fit_raw))
sink()

print("Done.")
