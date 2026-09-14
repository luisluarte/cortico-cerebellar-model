library(ggplot2)
library(gridExtra)

df_policy <- data.frame(
  Condition = factor(rep(c("Intact (Optimized)", "Acute Lesion", "Compensated Lesion"), each=2), 
                     levels=c("Intact (Optimized)", "Acute Lesion", "Compensated Lesion")),
  Feedback = factor(rep(c("Loss", "Win"), 3), levels=c("Loss", "Win")),
  P_Switch = c(0.449, 0.179, 0.534, 0.466, 0.460, 0.177)
)

df_wsls <- data.frame(
  Condition = factor(c("Intact (Optimized)", "Acute Lesion", "Compensated Lesion"), 
                     levels=c("Intact (Optimized)", "Acute Lesion", "Compensated Lesion")),
  WSLS = c(0.689, 0.534, 0.694)
)

p1 <- ggplot(df_policy, aes(x = Condition, y = P_Switch, fill = Feedback)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color="black") +
  scale_fill_manual(values = c("Loss" = "#e41a1c", "Win" = "#377eb8")) +
  theme_bw(base_size = 14) +
  labs(
    title = "A. Policy Breakdown (Marginal Probabilities)",
    subtitle = "Acute lesion collapses to ~50%. Compensated lesion mimics intact margins.",
    y = "Mean P(Switch)",
    x = ""
  ) +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 15, hjust = 1)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", alpha=0.5)

p2 <- ggplot(df_wsls, aes(x = Condition, y = WSLS, fill = Condition)) +
  geom_bar(stat = "identity", width = 0.5, color="black") +
  scale_fill_manual(values = c("Intact (Optimized)" = "#1b9e77", "Acute Lesion" = "grey50", "Compensated Lesion" = "#e7298a")) +
  theme_bw(base_size = 14) +
  labs(
    title = "B. Global Similarity to WSLS Policy",
    subtitle = "The Compensated cortex recovers an artificial rule-based concordance.",
    y = "P(Match WSLS Rule)",
    x = ""
  ) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 15, hjust = 1)) +
  coord_cartesian(ylim = c(0.4, 0.8)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", alpha=0.5)

p_final <- grid.arrange(p1, p2, ncol = 2, top = "The Tripartite Failure: Acute Collapse vs. Compensated Rigidity")
ggsave("Fig70_Tripartite_Policy.png", p_final, width = 11, height = 6, dpi = 300)
