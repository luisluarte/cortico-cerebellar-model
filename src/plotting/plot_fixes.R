library(ggplot2)
library(dplyr)
library(viridis)

pub_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

cat("Updating Figure 2A: RNN Calibration Diagram...\n")
ece_df <- readRDS("results/ECE_calibrated.rds")
calib_summary <- ece_df %>%
  mutate(bin_group = ntile(cal_prob, 10)) %>%
  group_by(bin_group) %>%
  summarize(
    mean_pred = mean(cal_prob),
    empirical_prob = mean(true_label),
    count = n()
  )
p2a <- ggplot(calib_summary, aes(x = mean_pred, y = empirical_prob)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(aes(size = count), color = "darkred", alpha = 0.7) +
  geom_line(color = "darkred", linewidth = 1) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "A. Teacher RNN Calibration (Platt Scaling)",
    x = "Predicted Probability (Teacher)",
    y = "Empirical Probability (Human)"
  ) + pub_theme
ggsave("results/Fig2A_Calibration.png", plot = p2a, width = 6, height = 5, dpi = 300)

cat("Updating Figure 2B: Biological Loss Landscape (Desaturated)...\n")
land_df <- readRDS("results/landscape_data.rds")
# Cap the maximum loss at the 90th percentile to prevent saturation from outliers
cap_val <- quantile(land_df$empirical_nll, 0.90, na.rm = TRUE)
land_df$empirical_nll_capped <- pmin(land_df$empirical_nll, cap_val)

p2b <- ggplot(land_df, aes(x = alpha_pc, y = kappa_cf, z = empirical_nll_capped)) +
  geom_contour_filled(bins = 20) +
  scale_fill_viridis_d(option = "mako", direction = -1) +
  labs(
    title = "B. Biological Loss Landscape",
    subtitle = "Purkinje Learning Rate vs. Climbing Fiber Error",
    x = expression(alpha[PC]~"(Purkinje Learning Rate)"),
    y = expression(kappa[CF]~"(Climbing Fiber Modulation)"),
    fill = "Loss"
  ) + pub_theme + theme(legend.position = "right")
ggsave("results/Fig2B_Landscape.png", plot = p2b, width = 8, height = 5, dpi = 300)

cat("Generating Figure 3: Bayesian Model Comparison (LOO-CV)...\n")
loo_dist <- readRDS("final_model/results/experiment1_distillation_loo.rds")
loo_emp <- readRDS("final_model/results/experiment234_empirical_loo.rds")

# Function to safely extract elpd_loo
get_elpd <- function(loo_obj, model_name, experiment) {
  # LOO objects have estimates array: row 1 is elpd_loo, col 1 is Estimate, col 2 is SE
  est <- loo_obj$estimates["elpd_loo", "Estimate"]
  se <- loo_obj$estimates["elpd_loo", "SE"]
  data.frame(Model = model_name, Experiment = experiment, ELPD = est, SE = se)
}

# Empirical Prediction Models (Experiment 2, 3, 4)
df_emp <- bind_rows(
  get_elpd(loo_emp$loo_w_emp, "WSLS", "Human Empirical Data"),
  get_elpd(loo_emp$loo_q_emp, "Q-Learning", "Human Empirical Data"),
  get_elpd(loo_emp$loo_bio_lock, "Cerebellar ODE (Locked)", "Human Empirical Data")
)

p3 <- ggplot(df_emp, aes(x = reorder(Model, ELPD), y = ELPD, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8, color="black") +
  geom_errorbar(aes(ymin = ELPD - 2*SE, ymax = ELPD + 2*SE), width = 0.2, size=1) +
  coord_flip() +
  scale_fill_viridis_d(option="plasma", begin=0.2, end=0.8) +
  labs(
    title = "Bayesian Predictive Performance (PSIS-LOO)",
    subtitle = "Expected Log Predictive Density (Higher is better)",
    x = "Cognitive Model",
    y = "ELPD (LOO-CV) +/- 2 SE"
  ) + pub_theme + theme(legend.position = "none")

ggsave("results/Fig3_Bayesian_LOO.png", plot = p3, width = 8, height = 5, dpi = 300)
cat("Bayesian plots generated!\n")
