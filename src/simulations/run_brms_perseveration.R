library(brms)
library(dplyr)

df <- readRDS("results/perseveration_df.rds")
df_loss <- df %>% filter(Loss_Streak > 0)
# Severe group where the effect resides
df_sev <- df_loss %>% filter(Delta_Beta < 0.35)

# P_Stay needs to be squeezed slightly away from exactly 0 or 1 for Beta regression
df_sev$P_Stay_Sq <- (df_sev$P_Stay * (nrow(df_sev) - 1) + 0.5) / nrow(df_sev)

# Convert to factors
df_sev$Condition <- factor(df_sev$Condition, levels=c("Lesioned", "Optimized"))
df_sev$Subject <- factor(df_sev$Subject)
# We can treat Loss_Streak as continuous or factor. Let's use continuous to match the LMM interaction term.
df_sev$Loss_Streak_Scale <- scale(df_sev$Loss_Streak)

cat("1. Fitting Bayesian Beta Regression...\n")
fit_brm <- brm(
  formula = P_Stay_Sq ~ Condition * Loss_Streak_Scale + (1 | Subject),
  data = df_sev,
  family = Beta(link = "logit"),
  chains = 2, cores = 2, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 42,
  backend = "cmdstanr"
)

saveRDS(fit_brm, "results/brms_perseveration_interaction.rds")
cat("2. Extracting Posteriors for Interaction...\n")

# Get conditional effects to plot later
ce <- conditional_effects(fit_brm, effects = "Loss_Streak_Scale:Condition")
saveRDS(ce, "results/brms_perseveration_ce.rds")

cat("Done.\n")
