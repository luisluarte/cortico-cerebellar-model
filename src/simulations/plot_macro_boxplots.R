library(ggplot2)
library(dplyr)
library(patchwork)

macro_df <- readRDS("results/macro_phenotypes.rds")

fill_scale <- scale_fill_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77"))

# Panel 1: CFI
p1 <- ggplot(macro_df, aes(x = Condition, y = CFI, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  geom_jitter(color = "black", alpha = 0.2, width = 0.15, size = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  fill_scale + theme_bw(base_size = 14) +
  labs(title = "1. Context-Adaptation (CFI)", y = "Δ P(Switch)", x = "") +
  theme(legend.position = "none")

# Panel 2: Variance
p2 <- ggplot(macro_df, aes(x = Condition, y = Var_P, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  geom_jitter(color = "black", alpha = 0.2, width = 0.15, size = 1.2) +
  fill_scale + theme_bw(base_size = 14) +
  labs(title = "2. Trajectory Variance", y = "Var(P)", x = "") +
  theme(legend.position = "none")

# Panel 3: AR1 (Sluggishness)
p3 <- ggplot(macro_df, aes(x = Condition, y = AR1_P, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.5) +
  geom_jitter(color = "black", alpha = 0.2, width = 0.15, size = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  fill_scale + theme_bw(base_size = 14) +
  labs(title = "3. Sluggishness (AR1)", y = "AR(1) Coefficient", x = "") +
  theme(legend.position = "none")

# Combine them side-by-side horizontally for easier reading
p_final <- p1 + p2 + p3 + plot_annotation(
  title = "Global Macro-Cognitive Phenotypes",
  subtitle = "Aggregated structural comparison (ignoring underlying lesion severity)",
  theme = theme(
    plot.title = element_text(size = 18, face = "bold"),
    plot.subtitle = element_text(size = 14, color="grey30")
  )
)

ggsave("Fig46_Macro_Boxplots.png", p_final, width=12, height=5, dpi=300)
cat("Done.\n")
