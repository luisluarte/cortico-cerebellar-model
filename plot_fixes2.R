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
  # REMOVED size=count to avoid misleading dot sizes for identical decile bins
  geom_point(color = "darkred", size = 4, alpha = 0.8) +
  geom_line(color = "darkred", linewidth = 1) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "A. Teacher RNN Calibration (Platt Scaling)",
    x = "Predicted Probability (Teacher)",
    y = "Empirical Probability (Human)"
  ) + pub_theme
ggsave("results/Fig2A_Calibration.png", plot = p2a, width = 6, height = 5, dpi = 300)

cat("Updating Figure 2B: Biological Loss Landscape (Fixed Platau)...\n")
land_df <- readRDS("results/landscape_data.rds")

# The data contains a massive plateau of 10.0 (penalty for invalid ODEs).
# We isolate the actual converging valley (Loss < 10) to map the contour!
land_df <- land_df %>%
  mutate(is_valid = empirical_nll < 10,
         nll_clean = ifelse(empirical_nll < 10, empirical_nll, NA))

p2b <- ggplot(land_df, aes(x = alpha_pc, y = kappa_cf)) +
  # Render the valid valley with viridis
  geom_tile(aes(fill = nll_clean), width=0.001, height=0.002) +
  scale_fill_viridis_c(option = "mako", direction = -1, 
                       na.value = "gray90", name = "Loss (NLL)") +
  # Overlay a subtle outline of the stable region
  geom_contour(aes(z = ifelse(is.na(nll_clean), 1, 0)), breaks = 0.5, color="black", linetype="dashed") +
  labs(
    title = "B. Biological Loss Landscape",
    subtitle = "Gray region = Penalty (Unstable ODE). Colored region = Convergent Basin",
    x = expression(alpha[PC]~"(Purkinje Learning Rate)"),
    y = expression(kappa[CF]~"(Climbing Fiber Modulation)")
  ) + pub_theme + theme(legend.position = "right")

ggsave("results/Fig2B_Landscape.png", plot = p2b, width = 8, height = 5, dpi = 300)

cat("Fixes applied!\n")
