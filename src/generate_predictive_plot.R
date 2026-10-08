library(ggplot2)
library(ggdist)
library(ggthemes)
library(dplyr)
library(tidyr)
library(nlme)

# Load data
subj_errs <- readRDS("../results/subj_errs_all.rds")

bio_bce <- rowMeans(sapply(subj_errs$bio, function(x) x[, "bce"]))
bio_kin <- rowMeans(sapply(subj_errs$bio, function(x) sqrt(x[, "kin"]))) 

wsls_bce <- rowMeans(sapply(subj_errs$wsls, function(x) x[, "bce"]))
wsls_kin <- rowMeans(sapply(subj_errs$wsls, function(x) sqrt(x[, "kin"])))

df <- data.frame(
  Subject = rep(subj_errs$subjs, 2),
  Model = rep(c("Guided Bayesian", "WSLS"), each = length(subj_errs$subjs)),
  Decision_BCE = c(bio_bce, wsls_bce),
  Kinematic_RMSE = c(bio_kin, wsls_kin)
)

df_long <- pivot_longer(df, cols = c(Decision_BCE, Kinematic_RMSE), names_to = "Metric", values_to = "Value")
df_long$Metric <- factor(df_long$Metric, levels = c("Decision_BCE", "Kinematic_RMSE"), labels = c("Decision Loss (BCE)", "RT RMSE (Kinematic)"))
df_long$Model <- factor(df_long$Model, levels = c("Guided Bayesian", "WSLS"))

# Fit LMER for each metric
m_bce <- lme(Value ~ Model, random = ~ 1 | Subject, data = df_long %>% filter(Metric == "Decision Loss (BCE)"))
p_val_bce <- summary(m_bce)$tTable[2, "p-value"]

m_kin <- lme(Value ~ Model, random = ~ 1 | Subject, data = df_long %>% filter(Metric == "RT RMSE (Kinematic)"))
p_val_kin <- summary(m_kin)$tTable[2, "p-value"]

# Get predicted values
df_long$Predicted <- NA
df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"] <- predict(m_bce)
df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"] <- predict(m_kin)

df_long$x_num <- as.numeric(df_long$Model)

okabe_ito <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")

# Brackets
df_segments <- data.frame(
  Metric = factor(c("Decision Loss (BCE)", "Decision Loss (BCE)", "Decision Loss (BCE)", 
                    "RT RMSE (Kinematic)", "RT RMSE (Kinematic)", "RT RMSE (Kinematic)"),
                  levels = c("Decision Loss (BCE)", "RT RMSE (Kinematic)")),
  x = c(1, 1, 2, 1, 1, 2),
  xend = c(2, 1, 2, 2, 1, 2),
  y = c(max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.1, 
        max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.05, 
        max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.05, 
        max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.05, 
        max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.025, 
        max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.025),
  yend = c(max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.1, 
           max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.1, 
           max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.1, 
           max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.05, 
           max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.05, 
           max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.05)
)

df_text <- data.frame(
  Metric = factor(c("Decision Loss (BCE)", "RT RMSE (Kinematic)"), levels = c("Decision Loss (BCE)", "RT RMSE (Kinematic)")),
  x = c(1.5, 1.5),
  y = c(max(df_long$Predicted[df_long$Metric == "Decision Loss (BCE)"]) + 0.1, 
        max(df_long$Predicted[df_long$Metric == "RT RMSE (Kinematic)"]) + 0.05),
  label = c(ifelse(p_val_bce < 0.001, "***", ifelse(p_val_bce < 0.01, "**", ifelse(p_val_bce < 0.05, "*", "ns"))),
            ifelse(p_val_kin < 0.001, "***", ifelse(p_val_kin < 0.01, "**", ifelse(p_val_kin < 0.05, "*", "ns"))))
)

p <- ggplot(df_long, aes(x = x_num, y = Predicted, fill = Model)) +
  geom_point(aes(x = x_num - 0.2), shape = 21, size = 3, alpha = 0.5, position = position_jitter(width = 0.05, height = 0, seed = 42), color = "black") +
  geom_boxplot(width = 0.12, outlier.shape = NA, color = "black", linewidth = 0.6, alpha = 0.8, aes(group = x_num)) +
  stat_halfeye(adjust = 2, trim = TRUE, density = "unbounded", width = 0.5, .width = c(0.66, 0.95), justification = -0.3, point_colour = NA, color = "black") +
  geom_segment(data = df_segments, aes(x=x, xend=xend, y=y, yend=yend), inherit.aes = FALSE, color="black", linewidth=0.6) +
  geom_text(data = df_text, aes(x=x, y=y, label=label), inherit.aes = FALSE, vjust = 0, size = 5) +
  scale_fill_manual(values = okabe_ito[2:5]) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_x_continuous(breaks = 1:2, labels = levels(df_long$Model)) +
  labs(x = NULL, y = "Predicted Error", title = "Predictive Accuracy Comparison") +
  theme_classic(base_size = 18) +
  theme(legend.position = "none", strip.background = element_blank())

ggsave("../presentations/plots/plot_guided_wsls_mock.png", p, width = 10, height = 4, dpi = 300)
