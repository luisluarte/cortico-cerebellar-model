library(brms)
library(emmeans)
library(tidybayes)
library(ggplot2)
library(dplyr)
library(tidyr)

fit_brm <- readRDS("results/brms_switch_model_calibrated.rds")

# Evaluate marginal means
emm <- emmeans(fit_brm, ~ Condition | State, epred = TRUE)

# Extract draws
draws <- gather_emmeans_draws(emm)

# Pivot wider so each MCMC draw has the Optimized and Lesioned absolute probability side-by-side
draws_wide <- draws %>%
  select(.draw, State, Condition, .value) %>%
  pivot_wider(names_from = Condition, values_from = .value) %>%
  mutate(
    # Calculate the exact paired difference in probability of staying (Perseveration)
    # P(Stay) = 1 - P(Switch)
    Opt_P_Stay = 1 - Optimized,
    Les_P_Stay = 1 - Lesioned,
    Diff_P_Stay = Opt_P_Stay - Les_P_Stay
  )

# Plot the distribution of the Paired Difference
p <- ggplot(draws_wide, aes(x = Diff_P_Stay, y = State, fill = State)) +
  stat_halfeye(alpha = 0.8, .width = c(0.80, 0.95), point_interval = "median_hdi") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  scale_fill_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Posterior Distribution of the Within-Subject Difference",
    subtitle = "Δ P(Stay): (Optimized minus Lesioned)\nNegative values indicate a reduction in Perseveration",
    x = "Paired Difference in P(Stay)",
    y = "Task Environment"
  ) +
  theme(
    legend.position = "none",
    axis.text.y = element_text(face = "bold")
  )

ggsave("Fig29_Difference_Posteriors.png", p, width = 10, height = 6, dpi = 300)
