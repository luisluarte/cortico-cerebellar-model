library(ggplot2)
library(ggthemes)
library(dplyr)

df <- data.frame(
  Model = rep(c("Guided Bayesian", "WSLS"), each=2),
  Metric = rep(c("Decision Brier Score", "RT RMSE (s)"), 2),
  Value = c(0.12, 0.25, 0.28, 0.45) # Mock values where Guided is better than WSLS
)

okabe_ito <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")

p <- ggplot(df, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = position_dodge(), color="black") +
  scale_fill_manual(values = okabe_ito) +
  labs(y = "Metric Value", title = "Predictive Accuracy Comparison", fill="") +
  theme_classic(base_size=18) +
  theme(legend.position="bottom")

ggsave("../presentations/plots/plot_guided_wsls_mock.png", p, width = 10, height = 4, dpi = 300)
