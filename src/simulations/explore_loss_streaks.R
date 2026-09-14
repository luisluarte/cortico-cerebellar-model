library(ggplot2)
library(dplyr)
library(lmerTest)

df <- readRDS("results/perseveration_df.rds")

# We only care about trials where the agent just lost (Loss_Streak >= 1)
df_loss <- df %>% filter(Loss_Streak > 0)

# Cap loss streak at 5 for plotting stability
df_loss$Loss_Streak_Cap <- ifelse(df_loss$Loss_Streak > 4, "5+", as.character(df_loss$Loss_Streak))

# Calculate empirical averages per Streak level
agg_df <- df_loss %>%
  group_by(Condition, Loss_Streak_Cap) %>%
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
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  theme_bw(base_size = 15) +
  labs(
    title = "Perseveration: Markovian vs Non-Markovian Contexts",
    subtitle = "Probability of staying with a choice after N consecutive losses",
    x = "Consecutive Losses on the Same Choice (Loss Streak)",
    y = "P(Stay)"
  ) +
  theme(legend.position = "bottom")

ggsave("Fig51_Loss_Streak_Exploration.png", p, width = 8, height = 6, dpi = 300)

# Quick frequentist test on the interaction: Condition * Loss_Streak
m1 <- lmer(P_Stay ~ Condition * Loss_Streak + (1|Subject), data=df_loss)
cat("\n--- Frequentist Test (Linear Mixed Model) ---\n")
print(anova(m1))
cat("\nDone.\n")
