library(brms)
library(bayesplot)
library(ggplot2)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")

pars_to_plot <- c(
  "b_ConditionOptimized",
  "b_StateReversal",
  "b_log_Delta_Beta",
  "b_ConditionOptimized:StateReversal",
  "b_ConditionOptimized:log_Delta_Beta",
  "b_StateReversal:log_Delta_Beta",
  "b_ConditionOptimized:StateReversal:log_Delta_Beta"
)

clean_labels <- c(
  "Optimized (Main Effect)",
  "Reversal (Main Effect)",
  "log(Delta Beta)",
  "Optimized : Reversal",
  "Optimized : log(Delta Beta)",
  "Reversal : log(Delta Beta)",
  "Opt : Rev : log(Delta Beta)"
)

p <- mcmc_areas(
  as.matrix(fit_brm),
  pars = rev(pars_to_plot), # reverse so main effects are at the top
  prob = 0.95, 
  prob_outer = 0.99,
  point_est = "median"
) + 
  scale_y_discrete(labels = rev(clean_labels)) +
  theme_bw(base_size = 14) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Bayesian Posterior Distributions of Calibrated Effects",
    subtitle = "Shaded regions highlight the 95% Credible Intervals (Log-Odds Scale)",
    x = "Posterior Estimate (Logit Scale)"
  ) +
  theme(axis.text.y = element_text(face="bold"))

ggsave("Fig24_Posterior_Main_Effects.png", p, width = 10, height = 6, dpi = 300)
