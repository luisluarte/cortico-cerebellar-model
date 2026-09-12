library(ggplot2)
library(dplyr)
library(ggpubr)

cat("Generating statistical boxplots for Landscape Metrics...\n")

df <- readRDS("results/landscape_comparison_stats.rds")

# Transform data to long format to plot both metrics
df_long <- df %>%
  tidyr::pivot_longer(cols = c("L_max", "Kappa"), names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = recode(Metric, 
                         L_max = "Lipschitz Smoothness (L)",
                         Kappa = "Condition Number (Kappa)"))

# We use a log10 scale because eigenvalues and condition numbers are heavy-tailed
p <- ggpaired(df_long, x = "Model", y = "Value",
              color = "Model", line.color = "gray", line.size = 0.4,
              palette = "jco", facet.by = "Metric", scales = "free_y") +
  stat_compare_means(method = "wilcox.test", paired = TRUE, 
                     label = "p.format", label.x = 1.35) +
  scale_y_log10() +
  labs(title = "Knowledge Distillation Smoothes the High-Dimensional Optimization Landscape",
       subtitle = "Wilcoxon Signed-Rank Test across N=50 participants",
       y = "Log10(Value)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        strip.background = element_rect(fill = "white", color = "black"),
        strip.text = element_text(face="bold"))

ggsave("results/Fig4_Landscape_Stats.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Boxplot saved to results/Fig4_Landscape_Stats.png\n")
