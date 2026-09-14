library(brms)
library(emmeans)
library(ggplot2)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")

# Extract the data to find quartiles of log_Delta_Beta
df <- fit_brm$data
q_logs <- quantile(df$log_Delta_Beta, probs = c(0.25, 0.5, 0.75))

# Evaluate Bayesian Emmeans!
emm_res <- emmeans(fit_brm, ~ Condition | State * log_Delta_Beta, 
                   at = list(log_Delta_Beta = q_logs), epred = TRUE)

df_emm <- as.data.frame(emm_res)

# Transform P(Switch) to P(Stay)
df_emm$emmean_stay <- 1 - df_emm$emmean
df_emm$lower.HPD_stay <- 1 - df_emm$upper.HPD
df_emm$upper.HPD_stay <- 1 - df_emm$lower.HPD

df_emm$Quartile <- factor(ifelse(df_emm$log_Delta_Beta == q_logs[1], "Q1 (Smallest Lesion)",
                          ifelse(df_emm$log_Delta_Beta == q_logs[2], "Q2 (Median Lesion)", "Q3 (Largest Lesion)")),
                          levels = c("Q1 (Smallest Lesion)", "Q2 (Median Lesion)", "Q3 (Largest Lesion)"))

p <- ggplot(df_emm, aes(x = Condition, y = emmean_stay, color = Condition)) +
  facet_grid(State ~ Quartile) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbar(aes(ymin = lower.HPD_stay, ymax = upper.HPD_stay), 
                position = position_dodge(width = 0.5), linewidth = 1.2, width = 0.2) +
  theme_bw(base_size = 14) +
  labs(title = "Bayesian Posterior Perseveration P(Stay | Loss) by Ablation Magnitude",
       y = "Calibrated P(Stay) Mean & 95% HPD") +
  scale_color_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77")) +
  theme(strip.text = element_text(face="bold"))

ggsave("Fig26_Bayesian_Emmeans_Stay.png", p, width=11, height=7, dpi=300)
