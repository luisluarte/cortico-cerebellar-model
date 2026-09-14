library(dplyr)
library(glmmTMB)

fit <- readRDS("results/brms_switch_model_calibrated.rds")
df <- fit$data

# Baseline frequentist model (matches brms logic)
m1 <- glmmTMB(P_Calibrated ~ Condition * State * log_Delta_Beta + (1 | Subject),
              data = df, family = beta_family(link="logit"))

cat("--- Model 1: Baseline 3-Way Interaction ---\n")
print(summary(m1)$coefficients$cond)

# What if we model the dispersion (phi) as varying by State?
# In glmmTMB, dispformula handles the precision parameter (phi).
m2 <- glmmTMB(P_Calibrated ~ Condition * State * log_Delta_Beta + (1 | Subject),
              dispformula = ~ State * Condition,
              data = df, family = beta_family(link="logit"))

cat("\n--- Model 2: Varying Dispersion by State & Condition ---\n")
print(summary(m2)$coefficients$cond)

# What if we isolate the Reversal state entirely to see its pure dynamics without being anchored by Stable?
df_rev <- df %>% filter(State == "Reversal")
m_rev <- glmmTMB(P_Calibrated ~ Condition * log_Delta_Beta + (1 | Subject),
                 data = df_rev, family = beta_family(link="logit"))
cat("\n--- Model 3: Isolated Reversal State Model ---\n")
print(summary(m_rev)$coefficients$cond)

# Let's also check if transforming log_Delta_Beta differently helps.
# What if we use a natural spline with 3 degrees of freedom just on the Reversal data?
library(splines)
m_rev_spline <- glmmTMB(P_Calibrated ~ Condition * ns(log_Delta_Beta, df=3) + (1 | Subject),
                 data = df_rev, family = beta_family(link="logit"))
cat("\n--- Model 4: Isolated Reversal with Spline ---\n")
print(summary(m_rev_spline)$coefficients$cond)

