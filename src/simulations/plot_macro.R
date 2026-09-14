library(ggplot2)
library(dplyr)
library(patchwork)

macro_df <- readRDS("results/macro_phenotypes.rds")
line_df <- readRDS("results/macro_phenotypes_lines.rds")
empirical_min <- min(macro_df$Delta_Beta)

col_scale <- scale_color_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77"))
fill_scale <- scale_fill_manual(values=c("Lesioned"="#d95f02", "Optimized"="#1b9e77"))

p1 <- ggplot(macro_df, aes(x=Delta_Beta, y=CFI, color=Condition)) +
  geom_point(alpha=0.25, size=2) + geom_line(data=line_df, linewidth=1.5) +
  geom_hline(yintercept=0, linetype="dashed") +
  col_scale + theme_bw(base_size=13) +
  labs(title="1. Context-Adaptation Index (CFI)", subtitle="Mean P(Switch | Reversal) - Mean P(Switch | Stable)", y="Δ P(Switch)", x="") +
  xlim(empirical_min, 1.0) + theme(legend.position="none")
  
p2 <- ggplot(macro_df, aes(x=Delta_Beta, y=Var_P, color=Condition)) +
  geom_point(alpha=0.25, size=2) + geom_line(data=line_df, linewidth=1.5) +
  col_scale + theme_bw(base_size=13) +
  labs(title="2. Global Trajectory Variance", subtitle="Total variance of P(Switch) across the session", y="Var(P)", x="") +
  xlim(empirical_min, 1.0) + theme(legend.position="none")
  
p3 <- ggplot(macro_df, aes(x=Delta_Beta, y=AR1_P, color=Condition)) +
  geom_point(alpha=0.25, size=2) + geom_line(data=line_df, linewidth=1.5) +
  geom_hline(yintercept=0, linetype="dashed") +
  col_scale + theme_bw(base_size=13) +
  labs(title="3. Non-Markovian Sluggishness (AR1)", subtitle="Lag-1 Autocorrelation of P(Switch) updating", y="AR(1) Coefficient", x=bquote("Thalamic Connection Scalar ("~beta[thal]~")")) +
  xlim(empirical_min, 1.0) + theme(legend.position="bottom")
  
p_final <- p1 / p2 / p3 + plot_annotation(
  title="Macro-Cognitive Phenotypes of Cerebellar Ablation", 
  subtitle="Revealing the cerebellum's true role in global structural flexibility",
  theme=theme(plot.title=element_text(size=18, face="bold"), plot.subtitle=element_text(size=14, color="grey30"))
)

ggsave("Fig45_Macro_Phenotypes.png", p_final, width=8, height=12, dpi=300)
cat("Done.\n")
