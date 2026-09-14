library(brms)
library(dplyr)
library(ggplot2)
library(gridExtra)

fit <- readRDS("results/brms_global_parametric.rds")

ldb_seq <- seq(-2.3, 0.0, length.out = 100)

nd_opt <- data.frame(Condition = "Optimized", log_Delta_Beta = ldb_seq)
nd_les <- data.frame(Condition = "Lesioned", log_Delta_Beta = ldb_seq)

cat("Extracting posterior predictions on the RESPONSE scale...\n")
draws_opt <- fitted(fit, newdata = nd_opt, re_formula = NA, summary = FALSE)
draws_les <- fitted(fit, newdata = nd_les, re_formula = NA, summary = FALSE)
draws_diff <- draws_opt - draws_les

sum_opt <- data.frame(
  Condition = "Optimized",
  log_Delta_Beta = ldb_seq,
  Delta_Beta = exp(ldb_seq),
  P_Median = apply(draws_opt, 2, median),
  Lower = apply(draws_opt, 2, quantile, probs = 0.025),
  Upper = apply(draws_opt, 2, quantile, probs = 0.975)
)

sum_les <- data.frame(
  Condition = "Lesioned",
  log_Delta_Beta = ldb_seq,
  Delta_Beta = exp(ldb_seq),
  P_Median = apply(draws_les, 2, median),
  Lower = apply(draws_les, 2, quantile, probs = 0.025),
  Upper = apply(draws_les, 2, quantile, probs = 0.975)
)

sum_abs <- rbind(sum_opt, sum_les)

sum_diff <- data.frame(
  log_Delta_Beta = ldb_seq,
  Delta_Beta = exp(ldb_seq),
  Diff_Median = apply(draws_diff, 2, median),
  Lower = apply(draws_diff, 2, quantile, probs = 0.025),
  Upper = apply(draws_diff, 2, quantile, probs = 0.975)
)

p_abs <- ggplot(sum_abs, aes(x = Delta_Beta, y = P_Median, color = Condition, fill = Condition)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_reverse() + 
  theme_bw(base_size = 14) +
  labs(
    title = "A. Global Cognitive Flexibility",
    subtitle = "Posterior expected probabilities",
    x = bquote("Structural Integrity (" ~ Delta*beta[thal] ~ ")  [Health -> Damage]"),
    y = "P(Switch | Loss)"
  ) +
  theme(legend.position = "bottom")

p_diff <- ggplot(sum_diff, aes(x = Delta_Beta, y = Diff_Median)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  geom_line(color = "blue", linewidth = 1.5) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "blue", alpha = 0.2) +
  scale_x_reverse() +
  theme_bw(base_size = 14) +
  labs(
    title = "B. The Cerebellar Correction",
    subtitle = bquote(Delta ~ "P(Switch) [Optimized - Lesioned]"),
    x = bquote("Structural Integrity (" ~ Delta*beta[thal] ~ ")  [Health -> Damage]"),
    y = "Change in P(Switch) provided by Cerebellum"
  )

p_final <- grid.arrange(p_abs, p_diff, ncol = 2, top = "The Cerebellum as a Biological Shock Absorber")

ggsave("Fig61_Global_Response_Natural.png", p_final, width = 12, height = 6, dpi = 300)
cat("Done.\n")
