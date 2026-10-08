library(ggplot2)
library(ggdist)
library(ggthemes)
library(dplyr)
library(tidyr)

okabe_ito <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")

# 1. NSGA-II Hypervolume Decay (3-Model Comparison)
df_hv <- read.csv("../data/gamma_hv_bootstrap_4d_strict.csv") %>% filter(Model %in% c("bio", "q", "wsls"))
df_hv$Model <- factor(df_hv$Model, levels=c("bio", "q", "wsls"), labels=c("Guided Bayesian", "Q-Learning", "WSLS"))

p_hv <- ggplot(df_hv, aes(x = Gamma, y = HV_mean, color = Model, fill = Model)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymin = HV_lwr, ymax = HV_upr), alpha = 0.2, color = NA) +
  scale_color_manual(values = okabe_ito) +
  scale_fill_manual(values = okabe_ito) +
  labs(x = expression(paste("Autonomy (", gamma, ")")), y = expression(paste("4D Hypervolume (", mu, ")")), title = "NSGA-II Performance Decay") +
  theme_classic(base_size = 18) +
  theme(legend.position = "bottom", legend.title = element_blank())

ggsave("../presentations/plots/plot_nsga2_hv_ggdist.png", p_hv, width=10, height=4, dpi=300)


# 2. PSIS-LOO Bar Plot with Faceted Pareto Warnings
df_loo <- data.frame(
  Model = c("Unconstrained\n(Flat Prior)", "Guided\n(sigma=e^-0.5)", "Zero-Shot\n(sigma=e^-1.5)", "WSLS", "Q-Learning"),
  ELPD = c(1712.0, 1548.6, 1511.4, 1212.7, 1200.4),
  SE = c(4.5, 4.9, 5.0, 5.2, 3.7),
  Pareto_K_Warnings = c(389, 0, 0, 0, 0)
)
df_loo$Model <- factor(df_loo$Model, levels=df_loo$Model)
df_long_loo <- pivot_longer(df_loo, cols = c(ELPD, Pareto_K_Warnings), names_to = "Metric", values_to = "Value")
df_long_loo$Metric <- factor(df_long_loo$Metric, levels = c("ELPD", "Pareto_K_Warnings"), labels = c("PSIS-LOO ELPD", "Pareto k > 0.7 Warnings"))

p_loo <- ggplot(df_long_loo, aes(x = Model, y = Value, fill = Model)) +
  geom_col(width = 0.6, color="black", size=0.8) +
  geom_errorbar(data = subset(df_long_loo, Metric == "PSIS-LOO ELPD"), 
                aes(ymin = Value - 1.96*SE, ymax = Value + 1.96*SE), width = 0.2, size=0.8) +
  scale_fill_manual(values = okabe_ito) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(y = "Metric Value", x = NULL, title = "Out-of-Sample Generalization") +
  theme_classic(base_size=16) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1), strip.background = element_blank())

ggsave("../presentations/plots/plot_psis_loo_ggdist.png", p_loo, width = 10, height = 4, dpi = 300)


# 3. Policy Perseverance (Entropy) - Smoother Kernel & Clearer Median
df_ent <- read.csv("../data/diff_entropy_ready.csv") %>% filter(Ablation_Pct %in% c(0, 90, 100))

p_ent <- ggplot(df_ent, aes(x = factor(Ablation_Pct), y = Total_Diff_Entropy, fill = factor(Ablation_Pct))) +
  stat_halfeye(adjust = 1.5, width = .6, .width = 0, justification = -.2, point_colour = NA) +
  geom_boxplot(width = .12, outlier.shape = NA, alpha=0.8, color="black", fatten=4) +
  geom_point(size = 3, shape = 21, position = position_jitter(width = .05, seed = 42), alpha = 0.7, color="black") +
  scale_fill_manual(values = okabe_ito[1:3]) +
  labs(x = "Thalamic Ablation (%)", y = "Differential Entropy (nats)", title = "Behavioral Rigidity") +
  theme_classic(base_size=18) + theme(legend.position="none")

ggsave("../presentations/plots/plot_entropy_ggdist.png", p_ent, width=10, height=4, dpi=300)

# 4. Readout Efficiency - Smoother Kernel & Clearer Median
df_eff <- read.csv("../data/readout_efficiency_df.csv") %>% filter(Ablation_Pct %in% c(0, 90, 100))

p_eff <- ggplot(df_eff, aes(x = factor(Ablation_Pct), y = Efficiency, fill = factor(Ablation_Pct))) +
  stat_halfeye(adjust = 1.5, width = .6, .width = 0, justification = -.2, point_colour = NA) +
  geom_boxplot(width = .12, outlier.shape = NA, alpha=0.8, color="black", fatten=4) +
  geom_point(size = 3, shape = 21, position = position_jitter(width = .05, seed = 42), alpha = 0.7, color="black") +
  scale_fill_manual(values = okabe_ito[c(4,5,6)]) +
  labs(x = "Thalamic Ablation (%)", y = expression(paste("Cortical Readout Efficiency (", R^2 / V, ")")), title = "Representational Collapse") +
  theme_classic(base_size=18) + theme(legend.position="none")

ggsave("../presentations/plots/plot_efficiency_ggdist.png", p_eff, width=10, height=4, dpi=300)
