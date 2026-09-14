library(ggplot2)
library(dplyr)
library(mgcv)

# Load the exact dataset used for the Spline models (which is strictly filtered to Loss_{t-1} trials)
fit <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
df <- fit$data

# Convert Switch to Stay
# (If raw P_Switch is present, use it. Otherwise use the P_Calibrated we fitted on)
if("P_Switch" %in% colnames(df)) {
  df$P_Stay <- 1 - df$P_Switch
  ylab_text <- "Raw P(Stay | Loss)"
} else {
  df$P_Stay <- 1 - df$P_Calibrated
  ylab_text <- "Calibrated P(Stay | Loss)"
}

# Bound to empirical limits
empirical_min <- min(df$Delta_Beta)
df <- df %>% filter(Delta_Beta <= 1.0)

# Build the scatter plot
p <- ggplot(df, aes(x = Delta_Beta, y = P_Stay, color = State)) +
  geom_point(alpha = 0.3, size = 1.5) +
  facet_wrap(~ Condition) +
  # Overlay a stiff GAM line (k=4) so we can see the geometric trend clearly
  geom_smooth(method = "gam", formula = y ~ s(x, k=4), color="black", linewidth=1.2, se=FALSE) +
  scale_color_manual(values = c("Stable" = "#377eb8", "Reversal" = "#e41a1c")) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25), limits=c(empirical_min, 1.0)) +
  theme_bw(base_size = 14) +
  labs(title = "Probability of Staying (Not Switching) after a Loss",
       subtitle = "P(Stay | Loss) mapped across the structural severity continuum",
       x = bquote("Thalamic Connection Scalar (" ~ beta[thal] ~ ")\n<-- 100% Ablation (Disconnected)                     Fully Intact (1.0) -->"),
       y = ylab_text) +
  theme(legend.position = "bottom", strip.text=element_text(face="bold"))
  
ggsave("Fig48_Raw_Stay_Loss.png", p, width=10, height=6, dpi=300)
cat("Done.\n")
