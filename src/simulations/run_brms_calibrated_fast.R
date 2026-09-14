library(brms)
library(dplyr)

# Load the mathematically stable calibrated probabilities
long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)

# Dropping the complex Condition*State interaction
# Note: brms requires the parametric main effect 'Condition' to be present 
# when defining separate smooths via 'by = Condition', otherwise the splines 
# are mathematically forced to share the exact same global mean intercept.
fast_formula <- bf(
  P_Calibrated ~ Condition + State + s(Delta_Beta, by = Condition) + (1 | Subject),
  phi ~ 1
)

print("Fitting Fast Calibrated Spline...")

fit_fast <- brm(
  formula = fast_formula,
  data = long_df,
  family = Beta(link = "logit", link_phi = "log"),
  chains = 4, cores = 4, 
  iter = 2000, warmup = 1000, # 1000 sampling draws per chain
  control = list(adapt_delta = 0.90, max_treedepth = 10),
  seed = 42
)

saveRDS(fit_fast, "results/brms_switch_model_calibrated_fast.rds")

sink("results/brms_switch_summary_calibrated_fast.txt")
print(summary(fit_fast))
sink()

print("Done.")
