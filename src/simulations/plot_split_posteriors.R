library(brms)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(patchwork)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")
draws <- as_draws_df(fit_brm)

# Calculate state-specific Bayesian posterior draws for the relevant mechanistic effects
# 1. Base behavioral rescue (Optimized vs Lesioned intercept difference)
stable_rescue <- draws$b_ConditionOptimized
reversal_rescue <- draws$b_ConditionOptimized + draws$`b_ConditionOptimized:StateReversal`

# 2. Scaling with Lesion Magnitude (Interaction with log_Delta_Beta)
stable_magnitude <- draws$`b_ConditionOptimized:log_Delta_Beta`
reversal_magnitude <- draws$`b_ConditionOptimized:log_Delta_Beta` + draws$`b_ConditionOptimized:StateReversal:log_Delta_Beta`

# Bind into matrices for plotting
mat_stable <- cbind(
  "Sensitivity to Magnitude\n(Opt. : log(Delta Beta))" = stable_magnitude,
  "Behavioral Rescue\n(Optimized vs Lesioned)" = stable_rescue
)

mat_reversal <- cbind(
  "Sensitivity to Magnitude\n(Opt. : log(Delta Beta))" = reversal_magnitude,
  "Behavioral Rescue\n(Optimized vs Lesioned)" = reversal_rescue
)

# Generate ridges
p1 <- mcmc_areas(mat_stable, prob = 0.95, prob_outer = 0.99, point_est = "median") + 
  geom_vline(xintercept = 0, linetype = "dashed", color = "#d95f02", linewidth = 1) +
  theme_bw(base_size = 14) +
  labs(title = "Stable State Environment", x = "") +
  theme(axis.text.y = element_text(face="bold"), plot.title = element_text(face="bold"))
  
p2 <- mcmc_areas(mat_reversal, prob = 0.95, prob_outer = 0.99, point_est = "median") + 
  geom_vline(xintercept = 0, linetype = "dashed", color = "#d95f02", linewidth = 1) +
  theme_bw(base_size = 14) +
  labs(title = "Reversal State Environment", x = "Bayesian Posterior Estimate (Log-Odds Scale)") +
  theme(axis.text.y = element_text(face="bold"), plot.title = element_text(face="bold"))

caption_text <- "Bayesian Posterior Distributions separated by task state. The 'Behavioral Rescue' represents the baseline \nincrease in switch probability for the Optimized network relative to the Lesioned baseline. The 'Sensitivity \nto Magnitude' represents the interaction slope, demonstrating that the degree of behavioral rescue \nscales strongly with the underlying cerebellar ablation magnitude in both environmental conditions. \nShaded regions denote the 95% Highest Posterior Density (HPD) intervals; the red dashed line \nmarks the null hypothesis (zero effect)."

p_final <- p1 / p2 + 
  plot_annotation(
    title = "State-Specific Posterior Distributions of Cerebellar Rescue Effects",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.caption = element_text(hjust = 0, size = 12, face = "italic", color="grey30", margin=margin(t=15))
    )
  )
  
ggsave("Fig25_State_Specific_Posteriors.png", p_final, width = 11, height = 8, dpi = 300)
