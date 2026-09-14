library(brms)
library(emmeans)
library(tidybayes)
library(ggplot2)
library(dplyr)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")

# We want the absolute marginal posterior probabilities.
# We will evaluate them across Condition and State (automatically marginalized at the mean log_Delta_Beta)
emm <- emmeans(fit_brm, ~ Condition | State, epred = TRUE)

# Extract the exact 4,000 posterior draws for each marginal mean!
draws <- gather_emmeans_draws(emm)

# Generate a beautiful raincloud / half-eye plot of the absolute probabilities
p <- ggplot(draws, aes(x = .value, y = Condition, fill = Condition)) +
  facet_wrap(~ State, ncol = 1) +
  stat_halfeye(alpha = 0.8, .width = c(0.80, 0.95), point_interval = "median_hdi") +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Absolute Posterior Distributions of P(Switch | Loss)",
    subtitle = "Marginal predicted probabilities (evaluated at mean ablation magnitude)",
    x = "Posterior Predicted Probability of Switching",
    y = ""
  ) +
  theme(
    strip.text = element_text(face = "bold", size = 15),
    legend.position = "none",
    axis.text.y = element_text(face = "bold")
  )

ggsave("Fig28_Absolute_Posteriors_Switch.png", p, width = 9, height = 7, dpi = 300)
