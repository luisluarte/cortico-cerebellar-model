library(dplyr)
library(lmerTest)
library(emmeans)

long_df_all <- readRDS("results/long_df_all.rds")
safe_cor <- function(x) { 
  s <- sd(x, na.rm=TRUE)
  if(!is.na(s) && length(x) > 3 && s > 0) cor(x[-length(x)], x[-1], use="complete.obs") else NA_real_ 
}

# --- TEST 1 (Severity) ---
test14_df <- long_df_all %>%
  group_by(Subject, Condition) %>%
  summarize(
    Delta_Beta = first(Delta_Beta),
    AR1_Cont = safe_cor(P_Switch),
    .groups = "drop"
  ) %>% filter(!is.na(AR1_Cont))
  
test14_df$Severity <- ifelse(test14_df$Delta_Beta < 0.35, "Severe", "Healthy")
test14_df$Subject <- as.factor(test14_df$Subject)
test14_df$Condition <- as.factor(test14_df$Condition)
test14_df$Severity <- as.factor(test14_df$Severity)

m1 <- lmer(AR1_Cont ~ Condition * Severity + (1 | Subject), data=test14_df)

cat("==================================================\n")
cat("TEST 1: AR1 by SEVERITY (Linear Mixed Model)\n")
cat("==================================================\n")
print(anova(m1))
cat("\n--- Simple Effects (Optimized vs Lesioned by Severity) ---\n")
emm1 <- emmeans(m1, ~ Condition | Severity)
print(pairs(emm1))

# --- TEST 3 (Epoch) ---
test3_df <- long_df_all %>%
  mutate(Epoch = case_when(
    Trial <= 33 ~ "Early",
    Trial > 67 ~ "Late",
    TRUE ~ "Mid"
  )) %>%
  filter(Epoch != "Mid") %>%
  group_by(Subject, Condition, Epoch) %>%
  summarize(
    AR1_Cont = safe_cor(P_Switch),
    .groups = "drop"
  ) %>% filter(!is.na(AR1_Cont))
  
test3_df$Subject <- as.factor(test3_df$Subject)
test3_df$Condition <- as.factor(test3_df$Condition)
test3_df$Epoch <- as.factor(test3_df$Epoch)

m3 <- lmer(AR1_Cont ~ Condition * Epoch + (1 | Subject), data=test3_df)

cat("\n==================================================\n")
cat("TEST 3: AR1 by EPOCH (Linear Mixed Model)\n")
cat("==================================================\n")
print(anova(m3))
cat("\n--- Simple Effects (Optimized vs Lesioned by Epoch) ---\n")
emm3 <- emmeans(m3, ~ Condition | Epoch)
print(pairs(emm3))
