library(brms)
library(dplyr)

df <- readRDS("results/brms_switch_model_calibrated.rds")[["data"]]

# Create a combined factor for the spline 'by' argument
df$Cond_State <- interaction(df$Condition, df$State)

cat("Fitting Reversal-Specific GAM Model...\n")
# P_Calibrated is already scaled between 0 and 1
fit_reversal_gam <- brm(
  formula = P_Calibrated ~ Condition * State + s(log_Delta_Beta, by = Cond_State) + (1 | Subject),
  data = df,
  family = Beta(link = "logit"),
  chains = 2, cores = 2, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 42,
  backend = "cmdstanr"
)

saveRDS(fit_reversal_gam, "results/brms_reversal_gam.rds")
cat("Model saved.\n")

ce <- conditional_effects(fit_reversal_gam, effects = "log_Delta_Beta:Cond_State")
saveRDS(ce, "results/brms_reversal_gam_ce.rds")
cat("Done.\n")
