library(brms)
library(emmeans)
library(tidybayes)
library(ggplot2)
library(dplyr)
library(tidyr)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")

df <- fit_brm$data
q_logs <- quantile(df$log_Delta_Beta, probs = c(0.25, 0.5, 0.75))

# We MUST evaluate the marginal difference conditional on the magnitude of the lesion!
emm <- emmeans(fit_brm, ~ Condition | State * log_Delta_Beta, 
               at = list(log_Delta_Beta = q_logs), epred = TRUE)
               
draws <- gather_emmeans_draws(emm)

draws_wide <- draws %>%
  select(.draw, State, Condition, log_Delta_Beta, .value) %>%
  pivot_wider(names_from = Condition, values_from = .value) %>%
  mutate(
    Opt_P_Stay = 1 - Optimized,
    Les_P_Stay = 1 - Lesioned,
    Diff_P_Stay = Opt_P_Stay - Les_P_Stay,
    Quartile = factor(ifelse(log_Delta_Beta == q_logs[1], "Q1 (Small Lesion)",
                      ifelse(log_Delta_Beta == q_logs[2], "Q2 (Median Lesion)", "Q3 (Large Lesion)")),
                      levels = c("Q1 (Small Lesion)", "Q2 (Median Lesion)", "Q3 (Large Lesion)"))
  )
  
p <- ggplot(draws_wide, aes(x = Diff_P_Stay, y = State, fill = State)) +
  facet_wrap(~ Quartile, ncol=1) +
  stat_halfeye(alpha = 0.8, .width = c(0.80, 0.95), point_interval = "median_hdi") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  scale_fill_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Within-Subject Paired Differences Conditional on Lesion Magnitude",
    subtitle = "Δ P(Stay): (Optimized minus Lesioned)\nNegative values = reduced perseveration. Notice how the effect shifts powerfully as lesions get larger!",
    x = "Paired Difference in P(Stay)",
    y = ""
  ) +
  theme(legend.position = "none", strip.text = element_text(face="bold", size=14))
  
ggsave("Fig30_Difference_Quartiles.png", p, width = 10, height = 9, dpi = 300)
