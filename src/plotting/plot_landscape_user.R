library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

pub_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

cat("Replicating exact user solution for Loss Landscape from generate_figures.R...\n")

data_p1 <- readRDS("results/landscape_data.rds") %>%
  as_tibble() %>%
  pivot_longer(
    cols = c(empirical_nll, rnn_nll)
  ) %>%
  group_by(name) %>%
  mutate(
    value = scale(value)
  ) %>%
  filter(alpha_pc > 0.0005)

p1 <- data_p1 %>%
  ggplot(aes(
    alpha_pc, kappa_cf,
    fill = exp(value)
  )) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "magma", name = "Exp(Z-Scored NLL)") +
  facet_wrap(~name, labeller = as_labeller(c("empirical_nll"="Empirical Loss", "rnn_nll"="RNN Distillation Loss"))) +
  scale_x_log10(expand = c(0, 0), labels = scales::label_log()) +
  scale_y_log10(expand = c(0, 0), labels = scales::label_log()) +
  labs(
    title = "B. Biological Loss Landscape",
    x = expression(alpha[PC]~"(Purkinje Learning Rate)"),
    y = expression(kappa[CF]~"(Climbing Fiber Modulation)")
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "white", color = "black"),
    strip.text = element_text(face="bold")
  )

ggsave("results/Fig2B_Landscape.png", plot = p1, width = 9, height = 5, dpi = 300)
cat("Done.\n")
