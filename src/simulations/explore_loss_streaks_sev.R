library(ggplot2)
library(dplyr)
library(lmerTest)

df <- readRDS("results/perseveration_df.rds")
df_loss <- df %>% filter(Loss_Streak > 0)
df_loss$Loss_Streak_Cap <- ifelse(df_loss$Loss_Streak > 4, "5+", as.character(df_loss$Loss_Streak))
df_loss$Severity <- ifelse(df_loss$Delta_Beta < 0.35, "Severe Damage (< 0.35)", "Healthy / Mild (> 0.35)")

agg_df <- df_loss %>%
  group_by(Condition, Loss_Streak_Cap, Severity) %>%
  summarize(
    Mean_P_Stay = mean(P_Stay, na.rm=TRUE),
    SE = sd(P_Stay, na.rm=TRUE) / sqrt(n()),
    .groups = "drop"
  )

agg_df$Loss_Streak_Cap <- factor(agg_df$Loss_Streak_Cap, levels = c("1", "2", "3", "4", "5+"))

p <- ggplot(agg_df, aes(x = Loss_Streak_Cap, y = Mean_P_Stay, color = Condition, group = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = Mean_P_Stay - SE, ymax = Mean_P_Stay + SE), width = 0.2, linewidth = 1) +
  facet_wrap(~ Severity) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Perseveration: The Cerebellar Failsafe",
    subtitle = "Probability of perseverating after consecutive losses",
    x = "Consecutive Losses on the Same Choice (Loss Streak)",
    y = "P(Stay)"
  ) +
  theme(legend.position = "bottom", strip.text=element_text(face="bold"))

ggsave("Fig52_Loss_Streak_Severity.png", p, width = 10, height = 6, dpi = 300)

cat("\n--- Frequentist Test (Severe Group) ---\n")
m_sev <- lmer(P_Stay ~ Condition * Loss_Streak + (1|Subject), data=df_loss %>% filter(Delta_Beta < 0.35))
print(anova(m_sev))
