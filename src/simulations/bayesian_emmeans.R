library(brms)
library(emmeans)
library(ggplot2)

fit_brm <- readRDS("results/brms_switch_model.rds")

# Extract the data to find quartiles of log_Delta_Beta
df <- fit_brm$data
q_logs <- quantile(df$log_Delta_Beta, probs = c(0.25, 0.5, 0.75))

# Evaluate Bayesian Emmeans!
emm_res <- emmeans(fit_brm, ~ Condition | State * log_Delta_Beta, 
                   at = list(log_Delta_Beta = q_logs), epred = TRUE)
pairs_res <- pairs(emm_res)

sink("results/brms_emmeans_summary.txt")
print(pairs_res)
sink()

df_emm <- as.data.frame(emm_res)
df_emm$Quartile <- factor(ifelse(df_emm$log_Delta_Beta == q_logs[1], "Q1",
                          ifelse(df_emm$log_Delta_Beta == q_logs[2], "Q2 (Median)", "Q3")),
                          levels = c("Q1", "Q2 (Median)", "Q3"))

p <- ggplot(df_emm, aes(x = Condition, y = emmean, color = Condition)) +
  facet_grid(State ~ Quartile) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbar(aes(ymin = lower.HPD, ymax = upper.HPD), 
                position = position_dodge(width = 0.5), width = 0.2, size=1) +
  theme_bw(base_size = 14) +
  labs(title = "Bayesian Posterior P(Switch) by log(Delta Beta) Quartiles",
       y = "P(Switch) Posterior Mean & 95% HPD") +
  scale_color_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77"))

ggsave("Fig22_Bayesian_Emmeans.png", p, width=10, height=6)
