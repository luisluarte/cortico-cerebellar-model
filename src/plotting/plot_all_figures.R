library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

# Common Theme for Publication
pub_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

cat("Generating Figure 1A: Pareto Front...\n")
pareto_df <- readRDS("results/distillation_2_fast_pareto.rds")
p1a <- ggplot(pareto_df, aes(x = Teacher_Penalty, y = NLL, color = Gamma)) +
  geom_point(alpha = 0.7, size = 3) +
  scale_color_viridis_c(option = "plasma") +
  facet_wrap(~Model, scales = "free_x") +
  labs(
    title = "A. Teacher-Student Pareto Front",
    x = "Teacher Distillation Loss (RMSE)",
    y = "Empirical Behavioral Loss (NLL)"
  ) + pub_theme
ggsave("results/Fig1A_Pareto.png", plot = p1a, width = 8, height = 5, dpi = 300)

cat("Generating Figure 1B: LNSO Out-of-Sample Performance...\n")
lnso_df <- readRDS("results/lnso_5fold_results.rds")
# Reshape for grouped barplot
lnso_long <- lnso_df %>%
  group_by(Model, Quartile) %>%
  summarize(
    Mean_Emp = mean(Test_BCE_Emp),
    Mean_Teacher = mean(Test_BCE_Teacher),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(Mean_Emp, Mean_Teacher), names_to = "Target", values_to = "BCE")

p1b <- ggplot(lnso_long, aes(x = Quartile, y = BCE, fill = Target)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  facet_wrap(~Model) +
  scale_fill_manual(values = c("Mean_Emp" = "dodgerblue4", "Mean_Teacher" = "darkorange3"),
                    labels = c("Empirical Data", "Teacher RNN")) +
  labs(
    title = "B. Zero-Shot LNSO Validation",
    x = "Distillation Quartile (Low to High Gamma)",
    y = "Out-of-Sample Cross-Entropy (BCE)"
  ) + pub_theme
ggsave("results/Fig1B_LNSO.png", plot = p1b, width = 8, height = 5, dpi = 300)

cat("Generating Figure 1C: Biological Parameter Stability...\n")
params_df <- readRDS("results/parameter_stability.rds")
bio_params <- params_df %>% filter(Model == "Bio") %>%
  select(Quartile, alpha_pc, lambda_pc, beta_thal, kappa_cf, alpha_gran, beta_gran, sigma2_diff) %>%
  pivot_longer(cols = -Quartile, names_to = "Parameter", values_to = "Value")

p1c <- ggplot(bio_params, aes(x = Quartile, y = Value, color = Parameter, group = Parameter)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_y_log10() +
  facet_wrap(~Parameter, scales = "free_y", ncol = 4) +
  labs(
    title = "C. Physiological Parameter Stability",
    x = "Distillation Quartile",
    y = "Parameter Value (Log Scale)"
  ) + pub_theme + theme(legend.position = "none")
ggsave("results/Fig1C_Stability.png", plot = p1c, width = 10, height = 5, dpi = 300)


cat("Generating Figure 2A: RNN Calibration Diagram...\n")
ece_df <- readRDS("results/ECE_calibrated.rds")
# Compute calibration bins
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
  geom_line(color = "darkred", size = 1) +
  labs(
    title = "A. Teacher RNN Calibration (Platt Scaling)",
    x = "Predicted Probability (Teacher)",
    y = "Empirical Probability (Human)"
  ) + pub_theme
ggsave("results/Fig2A_Calibration.png", plot = p2a, width = 6, height = 5, dpi = 300)

cat("Generating Figure 2B: Biological Loss Landscape...\n")
land_df <- readRDS("results/landscape_data.rds")
p2b <- ggplot(land_df, aes(x = alpha_pc, y = kappa_cf, z = empirical_nll)) +
  geom_contour_filled(bins = 15) +
  scale_fill_viridis_d(option = "mako", direction = -1) +
  labs(
    title = "B. Biological Loss Landscape",
    subtitle = "Purkinje Learning Rate vs. Climbing Fiber Error",
    x = expression(alpha[PC]~"(Purkinje Learning Rate)"),
    y = expression(kappa[CF]~"(Climbing Fiber Modulation)")
  ) + pub_theme + theme(legend.position = "right")
ggsave("results/Fig2B_Landscape.png", plot = p2b, width = 8, height = 5, dpi = 300)

cat("All manuscript figures generated successfully!\n")
