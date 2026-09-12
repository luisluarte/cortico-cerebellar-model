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

cat("Replicating EXACT user solution for Loss Landscape from generate_figures.R...\n")

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
  scale_fill_viridis_c(option = "magma") +
  facet_wrap(~name) +
  scale_x_continuous(expand = c(0, 0), labels = scales::label_log()) +
  scale_y_continuous(expand = c(0, 0), labels = scales::label_log()) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "white", color = "black"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    plot.margin = margin(t = 0.1, r = 0.1, b = 0.1, l = 0.1, unit = "cm"),
    aspect.ratio = 1
  )

ggsave("results/Fig2B_Landscape.png", plot = p1, width = 8, height = 4, dpi = 300)
cat("Done.\n")
