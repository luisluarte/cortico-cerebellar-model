library(brms)
library(ggplot2)

fit_spline <- readRDS("results/brms_switch_model_spline.rds")

# Extract the fitted conditional effects of the Spline
# This automatically evaluates the non-linear s() curve across Delta_Beta
ce <- conditional_effects(fit_spline, effects = "Delta_Beta:Condition", 
                          conditions = data.frame(State = c("Stable", "Reversal")),
                          re_formula = NA)

# We want to plot the exact empirical curve that broke the sampler
df_plot <- ce$`Delta_Beta:Condition`

p <- ggplot(df_plot, aes(x = Delta_Beta, y = estimate__, color = Condition, fill = Condition)) +
  facet_wrap(~ State) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_fill_manual(values = c("Lesioned" = "#d95f02", "Optimized" = "#1b9e77")) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_bw(base_size = 14) +
  labs(
    title = "Non-Linear Spline Model (GAM) Fits",
    subtitle = "WARNING: Model exhibits 27 Divergent Transitions (Mathematically Invalid)",
    x = "Thalamic Connection Strength (Raw Delta Beta)",
    y = "Predicted P(Switch | Loss)"
  ) +
  theme(strip.text = element_text(face="bold"), legend.position="bottom")

ggsave("Fig34_Spline_Pathology.png", p, width = 10, height = 6, dpi = 300)
