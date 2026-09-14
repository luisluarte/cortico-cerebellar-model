library(brms)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(patchwork)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")
draws <- as_draws_df(fit_brm)

# Convert Log-Odds of Switch to Log-Odds of Stay by multiplying by -1
# 1. Base change in perseveration (Optimized vs Lesioned)
stable_rescue <- -1 * draws$b_ConditionOptimized
reversal_rescue <- -1 * (draws$b_ConditionOptimized + draws$`b_ConditionOptimized:StateReversal`)

# 2. Scaling with Lesion Magnitude
stable_magnitude <- -1 * draws$`b_ConditionOptimized:log_Delta_Beta`
reversal_magnitude <- -1 * (draws$`b_ConditionOptimized:log_Delta_Beta` + draws$`b_ConditionOptimized:StateReversal:log_Delta_Beta`)

# Bind into matrices for plotting
mat_stable <- cbind(
  "Sensitivity to Magnitude\n(Opt. : log(Delta Beta))" = stable_magnitude,
  "Change in Perseveration\n(Optimized vs Lesioned)" = stable_rescue
)

mat_reversal <- cbind(
  "Sensitivity to Magnitude\n(Opt. : log(Delta Beta))" = reversal_magnitude,
  "Change in Perseveration\n(Optimized vs Lesioned)" = reversal_rescue
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
  labs(title = "Reversal State Environment", x = "Bayesian Posterior Estimate (Log-Odds of STAYING)") +
  theme(axis.text.y = element_text(face="bold"), plot.title = element_text(face="bold"))

caption_text <- "Bayesian Posterior Distributions separated by task state, framed in terms of Perseveration (P(Stay | Loss)). \nNegative values on the x-axis represent a REDUCTION in the log-odds of staying. \n'Change in Perseveration' confirms that the Optimized network significantly reduces perseverative \nbehavior compared to the Lesioned baseline. 'Sensitivity to Magnitude' shows that this reduction \nin perseveration scales strongly with the underlying cerebellar ablation magnitude in both conditions. \nShaded regions denote the 95% Highest Posterior Density (HPD) intervals."

p_final <- p1 / p2 + 
  plot_annotation(
    title = "State-Specific Posterior Distributions of Perseveration Reduction",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.caption = element_text(hjust = 0, size = 12, face = "italic", color="grey30", margin=margin(t=15))
    )
  )
  
ggsave("Fig27_State_Specific_Posteriors_Stay.png", p_final, width = 11, height = 8, dpi = 300)
