library(ggplot2)
library(dplyr)

long_df <- readRDS("results/brms_switch_model_calibrated.rds")$data
long_df$Delta_Beta <- exp(long_df$log_Delta_Beta)

# Summarize the raw empirical data in bins
binned_df <- long_df %>%
  mutate(bin = cut(Delta_Beta, breaks=seq(0, 1.0, 0.05))) %>%
  group_by(bin, Condition) %>%
  summarize(P_Mean = mean(P_Calibrated), .groups="drop")

p <- ggplot(binned_df, aes(x=bin, y=P_Mean, fill=Condition)) +
  geom_bar(stat="identity", position="dodge") +
  theme_bw() + theme(axis.text.x = element_text(angle=45, hjust=1))
ggsave("Fig40_Raw_Bins.png", p, width=10, height=5)
