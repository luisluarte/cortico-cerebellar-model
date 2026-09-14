library(brms)
library(dplyr)

long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)

# The user's hypothesis: The state-specific precision broke the sampler. 
# We revert to a global phi ~ 1, but keep the splines.
spline_formula <- bf(
  P_Calibrated ~ Condition * State + s(Delta_Beta, by = Condition) + (1 | Subject),
  phi ~ 1
)

print("Fitting Non-Linear Spline Model with GLOBAL Precision...")

fit_spline <- brm(
  formula = spline_formula,
  data = long_df,
  family = Beta(link = "logit", link_phi = "log"),
  chains = 4, cores = 4, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 12), # Max geometric safety
  seed = 42
)

saveRDS(fit_spline, "results/brms_switch_model_spline_global.rds")

sink("results/brms_switch_summary_spline_global.txt")
print(summary(fit_spline))
sink()

print("Done.")
