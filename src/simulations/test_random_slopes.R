library(glmmTMB)
df <- readRDS('results/brms_switch_model_calibrated.rds')[['data']]

cat("--- Random Intercept Only ---\n")
m1 <- glmmTMB(P_Calibrated ~ Condition * State * log_Delta_Beta + (1 | Subject), data = df, family = beta_family(link='logit'))
print(summary(m1)$coefficients$cond)

cat("\n--- Random Slope for Condition ---\n")
m2 <- glmmTMB(P_Calibrated ~ Condition * State * log_Delta_Beta + (Condition | Subject), data = df, family = beta_family(link='logit'))
print(summary(m2)$coefficients$cond)

cat("\n--- Random Slope for Condition + Dispformula ---\n")
m3 <- glmmTMB(P_Calibrated ~ Condition * State * log_Delta_Beta + (Condition | Subject), dispformula = ~ State, data = df, family = beta_family(link='logit'))
print(summary(m3)$coefficients$cond)
