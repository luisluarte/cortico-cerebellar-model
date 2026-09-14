library(dplyr)
library(ggplot2)
library(gridExtra)

df <- readRDS("results/reversal_df.rds")

# F == 1 (Win) -> WSLS says Stay (1 - P_Switch)
# F == 0 (Loss) -> WSLS says Switch (P_Switch)
df$WSLS_Match <- ifelse(df$F == 1, 1 - df$P_Switch, df$P_Switch)

wsls_subj <- df %>%
  group_by(Subject, Condition, log_Delta_Beta) %>%
  summarize(
    Mean_WSLS = mean(WSLS_Match),
    .groups = "drop"
  )
wsls_subj$Delta_Beta <- exp(wsls_subj$log_Delta_Beta)

cat("--- WSLS Concordance Summary ---\n")
print(wsls_subj %>% group_by(Condition) %>% summarize(Grand_Mean = mean(Mean_WSLS)))

p_box <- ggplot(wsls_subj, aes(x = Condition, y = Mean_WSLS, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "A. Collapse into Markovian Policy",
    subtitle = "Similarity to Win-Stay, Lose-Shift (WSLS)",
    y = "Probability of Matching WSLS Rule",
    x = ""
  ) +
  theme(legend.position = "none")

p_scatter <- ggplot(wsls_subj, aes(x = Delta_Beta, y = Mean_WSLS, color = Condition, fill=Condition)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2, linewidth = 1.5) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_reverse() +
  theme_bw(base_size = 15) +
  labs(
    title = "B. Structural Vulnerability to WSLS",
    subtitle = "Degradation forces the cortex into hyper-reactivity",
    x = bquote("Structural Integrity (" ~ Delta*beta[thal] ~ ") [Health -> Damage]"),
    y = "Probability of Matching WSLS Rule"
  ) +
  theme(legend.position = "bottom")

p_final <- grid.arrange(p_box, p_scatter, ncol = 2, top = "The Cerebellum Prevents Hyper-Reactive Thrashing")
ggsave("Fig68_WSLS_Similarity.png", p_final, width = 12, height = 6, dpi = 300)
cat("Done.\n")
