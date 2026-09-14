library(dplyr)
library(ggplot2)
library(brms)

cat("1. Building Markovian vs Non-Markovian Metrics Dataset...\n")
df <- readRDS("results/perseveration_df.rds")

# We want the continuous sequence of probabilities to measure AR1
safe_cor <- function(x) { 
  s <- sd(x, na.rm=TRUE)
  if(!is.na(s) && length(x) > 3 && s > 0) cor(x[-length(x)], x[-1], use="complete.obs") else NA_real_ 
}

macro_df <- df %>%
  group_by(Subject, Condition) %>%
  summarize(
    Delta_Beta = first(Delta_Beta),
    # Markovian: Overall reactivity to a loss
    Mean_P_Switch = mean(P_Switch, na.rm=TRUE),
    # Non-Markovian: Sluggishness / stickiness across time
    AR1 = safe_cor(P_Switch),
    .groups = "drop"
  ) %>%
  filter(!is.na(AR1) & !is.na(Mean_P_Switch))

# Bound to empirical limits
macro_df <- macro_df %>% filter(Delta_Beta <= 1.0)
macro_df$Subject <- factor(macro_df$Subject)
macro_df$Condition <- factor(macro_df$Condition, levels=c("Lesioned", "Optimized"))

cat("2. Fitting Bayesian Models (brms)...\n")

# Model 1: Markovian Reactivity (Beta regression)
# Squeeze slightly away from 0/1
macro_df$P_Switch_Sq <- (macro_df$Mean_P_Switch * (nrow(macro_df) - 1) + 0.5) / nrow(macro_df)
fit_markovian <- brm(
  formula = P_Switch_Sq ~ Condition + s(Delta_Beta, by=Condition) + (1 | Subject),
  data = macro_df,
  family = Beta(link = "logit"),
  chains = 2, cores = 2, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 42,
  backend = "cmdstanr",
  file = "results/brms_markovian_final"
)

# Model 2: Non-Markovian Perseveration (Gaussian regression for AR1)
fit_non_markovian <- brm(
  formula = AR1 ~ Condition + s(Delta_Beta, by=Condition) + (1 | Subject),
  data = macro_df,
  family = gaussian(),
  chains = 2, cores = 2, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 42,
  backend = "cmdstanr",
  file = "results/brms_non_markovian_final"
)

cat("3. Extracting and Plotting Conditional Effects...\n")
ce_markovian <- conditional_effects(fit_markovian, effects="Delta_Beta:Condition")
ce_non_markovian <- conditional_effects(fit_non_markovian, effects="Delta_Beta:Condition")

saveRDS(list(markovian = ce_markovian[[1]], non_markovian = ce_non_markovian[[1]]), "results/final_brms_effects.rds")
cat("Done.\n")
