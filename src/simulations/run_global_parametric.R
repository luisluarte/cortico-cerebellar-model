library(brms)
library(dplyr)
library(ggplot2)
library(bayesplot)

df <- readRDS("results/brms_switch_model_calibrated.rds")[["data"]]

cat("Fitting Global Parametric Model (Ignoring State)...\n")
fit_global <- brm(
  formula = P_Calibrated ~ Condition * log_Delta_Beta + (1 | Subject),
  data = df,
  family = Beta(link = "logit"),
  chains = 4, cores = 4, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 42,
  backend = "cmdstanr"
)

saveRDS(fit_global, "results/brms_global_parametric.rds")

# Extract posteriors for main effects
pars_to_plot <- c(
  "b_ConditionOptimized",
  "b_log_Delta_Beta",
  "b_ConditionOptimized:log_Delta_Beta"
)

clean_labels <- c(
  "Optimized (Main Effect)",
  "log(Delta Beta)",
  "Optimized : log(Delta Beta)"
)

p_post <- mcmc_areas(
  as.matrix(fit_global),
  pars = rev(pars_to_plot),
  prob = 0.95, 
  prob_outer = 0.99,
  point_est = "median"
) + 
  scale_y_discrete(labels = rev(clean_labels)) +
  theme_bw(base_size = 14) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Bayesian Posteriors: Global Task Performance",
    subtitle = "Model aggregates all trials (Stable & Reversal)",
    x = "Posterior Estimate (Logit Scale)"
  ) +
  theme(axis.text.y = element_text(face="bold"))

ggsave("Fig58_Global_Posteriors.png", p_post, width = 8, height = 4, dpi = 300)

# Extract Conditional Effects
ce <- conditional_effects(fit_global, effects = "log_Delta_Beta:Condition")

p_ce <- plot(ce, plot = FALSE)[[1]] +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 14) +
  labs(
    title = "Cerebellar Buffering Across the Entire Task",
    x = bquote("Structural Integrity (" ~ log(Delta * beta) ~ ")"),
    y = "Calibrated P(Switch)"
  ) +
  theme(legend.position = "bottom")

ggsave("Fig59_Global_Conditional_Effects.png", p_ce, width = 8, height = 6, dpi = 300)

cat("Done.\n")
