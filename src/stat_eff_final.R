
library(lme4)
library(emmeans)
library(effectsize)

df <- read.csv("results/action_efficiency_90_df.csv")
print(aggregate(cbind(R2, Volatility, Efficiency) ~ Condition, data=df, mean))

df$Condition_F <- factor(df$Condition, levels=c("Intact", "Lesion_90", "Lesion_100"))

cat("\n--- LMM for Action Efficiency ---\n")
model <- lmer(Efficiency ~ Condition_F + (1 | Subject), data = df)
print(anova(model))

em <- emmeans(model, ~ Condition_F)
contrasts <- contrast(em, method="pairwise")
print(summary(contrasts))

cat("\n--- Effect Size (Cohens d) ---\n")
s_contrasts <- summary(contrasts)
d_90 <- t_to_d(t = s_contrasts$t.ratio[1], df_error = s_contrasts$df[1])
d_100 <- t_to_d(t = s_contrasts$t.ratio[2], df_error = s_contrasts$df[2])
print(d_90)
print(d_100)

cat("\nDone.\n")

