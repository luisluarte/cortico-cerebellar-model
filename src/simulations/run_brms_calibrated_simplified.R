library(brms)
library(dplyr)

long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)

# Simplified structural architecture: 
# Main effects for Condition and State, NO interaction.
# Global precision phi ~ 1.
simplified_formula <- bf(
  P_Calibrated ~ Condition + State + s(Delta_Beta, by = Condition) + (1 | Subject),
  phi ~ 1
)

print("Fitting Simplified Calibrated Spline (adapt_delta = 0.99)...")

fit_simplified <- brm(
  formula = simplified_formula,
  data = long_df,
  family = Beta(link = "logit", link_phi = "log"),
  chains = 4, cores = 4, 
  iter = 2000, warmup = 1000, 
  control = list(adapt_delta = 0.99, max_treedepth = 12),
  seed = 42
)

saveRDS(fit_simplified, "results/brms_switch_model_calibrated_simplified.rds")

sink("results/brms_switch_summary_calibrated_simplified.txt")
print(summary(fit_simplified))
sink()

print("Done.")
