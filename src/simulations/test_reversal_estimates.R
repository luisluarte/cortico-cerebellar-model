library(dplyr)
library(glmmTMB)
library(splines)

df <- readRDS("results/reversal_df.rds")

# We ONLY care about when the network gets negative feedback!
# A reversal only matters if you actually LOSE.
df_loss <- df %>% filter(F == 0)

cat("--- Summary of Loss Data ---\n")
print(table(df_loss$State, df_loss$Condition))

# Let's isolate the Reversal state (with loss) to test our exact estimates
df_rev <- df_loss %>% filter(State == "Reversal")

m_rev <- glmmTMB(P_Switch_Sq ~ Condition * log_Delta_Beta + (1 | Subject),
                 data = df_rev, family = beta_family(link="logit"))
cat("\n--- Model 1: Isolated Reversal State (Losses Only) ---\n")
print(summary(m_rev)$coefficients$cond)

# What if we model dispersion by Condition?
m_rev_disp <- glmmTMB(P_Switch_Sq ~ Condition * log_Delta_Beta + (1 | Subject),
                      dispformula = ~ Condition,
                      data = df_rev, family = beta_family(link="logit"))
cat("\n--- Model 2: Isolated Reversal (Losses Only) with Varying Dispersion ---\n")
print(summary(m_rev_disp)$coefficients$cond)

# Full interaction model (Stable vs Reversal) on LOSSES ONLY
m_full <- glmmTMB(P_Switch_Sq ~ Condition * State * log_Delta_Beta + (1 | Subject),
                  dispformula = ~ State * Condition,
                  data = df_loss, family = beta_family(link="logit"))
cat("\n--- Model 3: Full Interaction (Losses Only, Varying Dispersion) ---\n")
print(summary(m_full)$coefficients$cond)

