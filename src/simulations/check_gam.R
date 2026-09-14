library(mgcv)
library(dplyr)
library(emmeans)
library(ggplot2)

long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)
long_df$Subject <- as.factor(long_df$Subject)

# Fit a frequentist GAM to instantly check shapes
fit_gam_10 <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=10) + s(Subject, bs="re"), data=long_df, family=betar(link="logit"))
fit_gam_4  <- bam(P_Calibrated ~ Condition + State + s(Delta_Beta, by=Condition, k=4)  + s(Subject, bs="re"), data=long_df, family=betar(link="logit"))

grid_beta <- seq(min(long_df$Delta_Beta), 1.0, length.out = 150)

get_diff <- function(fit) {
  emm <- emmeans(fit, ~ Condition | Delta_Beta, at = list(Delta_Beta = grid_beta), epred=TRUE)
  diff_df <- as.data.frame(pairs(emm, by="Delta_Beta", reverse=TRUE))
  return(diff_df)
}

d10 <- get_diff(fit_gam_10)
d4 <- get_diff(fit_gam_4)

d10$Model <- "k=10 (Flexible)"
d4$Model <- "k=4 (Stiff)"

df_plot <- rbind(d10, d4)

p <- ggplot(df_plot, aes(x=Delta_Beta, y=estimate, color=Model)) +
  geom_hline(yintercept=0, linetype="dashed") +
  geom_line(linewidth=1.5) +
  theme_bw() + labs(title="GAM Spline Basis Comparison")
ggsave("Fig39_GAM_Check.png", p, width=7, height=5)
