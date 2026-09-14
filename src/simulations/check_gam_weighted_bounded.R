library(mgcv)
library(dplyr)
library(emmeans)
library(ggplot2)

# Use the data directly from the pulled RDS
fit_spline <- readRDS("results/brms_switch_model_calibrated_simplified.rds")
long_df <- fit_spline$data

# Ensure Subject is a factor for mgcv random effects
long_df$Subject <- as.factor(long_df$Subject)

# 1. Compute density-based weights
dens <- density(long_df$Delta_Beta)
long_df$density_weight <- approx(dens$x, dens$y, xout = long_df$Delta_Beta)$y
long_df$density_weight <- long_df$density_weight / mean(long_df$density_weight)

# Fit models
fit_10_unw <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=10) + s(Subject, bs="re"), 
                  data=long_df, family=betar(link="logit"))
fit_4_unw <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=4) + s(Subject, bs="re"), 
                 data=long_df, family=betar(link="logit"))
fit_4_w <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=4) + s(Subject, bs="re"), 
               data=long_df, weights=density_weight, family=betar(link="logit"))
fit_3_w <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=3) + s(Subject, bs="re"), 
               data=long_df, weights=density_weight, family=betar(link="logit"))

# STRICT BOUNDS: Only predict up to 1.0 (Fully Intact)
# The previous plot went all the way to 2.68 because of outliers in the generative distribution!
empirical_min <- min(long_df$Delta_Beta)
grid_beta <- seq(empirical_min, 1.0, length.out = 150)

get_diff <- function(fit, name) {
  emm <- emmeans(fit, ~ Condition | Delta_Beta, at = list(Delta_Beta = grid_beta), epred=TRUE)
  diff_df <- as.data.frame(pairs(emm, by="Delta_Beta", reverse=TRUE))
  diff_df$Model <- name
  return(diff_df)
}

d1 <- get_diff(fit_10_unw, "1. k=10 (Unweighted)")
d2 <- get_diff(fit_4_unw, "2. k=4 (Unweighted)")
d3 <- get_diff(fit_4_w, "3. k=4 (Density-Weighted)")
d4 <- get_diff(fit_3_w, "4. k=3 (Density-Weighted)")

df_plot <- rbind(d1, d2, d3, d4)

p <- ggplot(df_plot, aes(x=Delta_Beta, y=estimate, color=Model)) +
  geom_hline(yintercept=0, linetype="dashed") +
  geom_line(linewidth=1.5, alpha=0.9) +
  # Adding a rug plot so you can explicitly see exactly where the data lives!
  geom_rug(data = long_df %>% filter(Delta_Beta <= 1.0), aes(x = Delta_Beta), inherit.aes = FALSE, alpha = 0.05, sides="b") +
  scale_x_continuous(breaks = seq(0, 1.0, 0.25), limits = c(empirical_min, 1.0)) +
  scale_color_viridis_d(option="turbo") +
  theme_bw(base_size=14) + 
  labs(title="GAM Spline Basis & Density Weighting Comparison",
       subtitle="Restricted strictly to the biological interval [Min, 1.0]",
       y="Difference (Optimized - Lesioned)", 
       x = bquote("Thalamic Connection Scalar (" ~ beta[thal] ~ ")\n<-- 100% Ablation (Disconnected)                     Fully Intact (1.0) -->")) +
  theme(legend.position = "bottom", legend.direction="vertical")
  
ggsave("Fig42_GAM_Weighted_Bounded.png", p, width=8, height=8, dpi=300)
