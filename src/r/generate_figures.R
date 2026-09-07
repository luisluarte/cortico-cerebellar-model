# setup ----
# lib path
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

local({
  r <- getOption("repos")
  r["CRAN"] <- "https://cloud.r-project.org"
  options(repos = r)
})

# install libs
cat("##########################################################################")
cat("\ninstalling libs...\n")
cat("##########################################################################\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman", lib = local_lib)
}

cat("##########################################################################")
cat("\nloading libs...\n")
cat("##########################################################################\n")
pacman::p_load(
  tidyverse,
  ggplot2,
  ggpubr,
  this.path,
  scales
)

setwd(here())

# custom theme
custom_colors <- c(
  "Parameter:15067"  = "#2a9d7a", # Teal / green
  "Parameter:30299"  = "#7572b8", # Slate purple / blue
  "Parameter:45083"  = "#73a027", # Olive green
  "Parameter:105275" = "#a6782b", # Warm brown / gold
  "Parameter:155803" = "#666666" # Muted grey
)

# Custom theme
theme_matplotlib_log <- function(base_size = 13, base_family = "sans") {
  theme_pubr(base_size = base_size, base_family = base_family) +
    theme(
      # Plot & Panel backgrounds
      plot.background      = element_rect(fill = "white", color = NA),
      panel.background     = element_rect(fill = "white", color = NA),

      # Complete outer bounding box
      panel.border         = element_rect(color = "black", fill = NA, linewidth = 0.8),

      # Solid major gridlines
      panel.grid.major     = element_line(color = "#cccccc", linewidth = 0.6),

      # Dashed minor gridlines (gives the vertical dashed log markers seen in the image)
      panel.grid.minor.x   = element_line(color = "#bfbfbf", linetype = "dashed", linewidth = 0.5),
      panel.grid.minor.y   = element_blank(),

      # Black axis ticks pointing outwards
      axis.ticks           = element_line(color = "black", linewidth = 0.6),
      axis.ticks.length    = unit(4, "pt"),

      # Typography
      plot.title           = element_text(size = rel(1.2), hjust = 0.5, face = "plain", color = "black"),
      axis.title.x         = element_text(size = rel(1.15), color = "black", margin = margin(t = 6)),
      axis.title.y         = element_text(size = rel(1.2), color = "black", margin = margin(r = 6)),
      axis.text            = element_text(size = rel(1.05), color = "black"),

      # Enclosed top-right legend box
      legend.position      = c(0.74, 0.74),
      legend.title         = element_blank(),
      legend.text          = element_text(size = rel(1.0), face = "plain", color = "black"),
      legend.background    = element_rect(fill = "white", color = "#d3d3d3", linewidth = 0.5),
      legend.key           = element_rect(fill = "white", color = NA),
      legend.key.width     = unit(20, "pt"),
      legend.spacing.y     = unit(2, "pt")
    )
}

# plots -------------------------------------------------------------------


## likelihood landscape ----------------------------------------------------

data_p1 <- read_rds("../../results/landscape_data.rds") %>%
  as_tibble() %>%
  pivot_longer(
    cols = c(empirical_nll, rnn_nll)
  ) %>%
  group_by(name) %>%
  mutate(
    value = scale(value)
  ) %>%
  filter(alpha_pc > 0.0005)

scale_factor <- 3
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
  theme_matplotlib_log() +
  theme(
    legend.position = "none",
    text = element_text(size = scale_factor * 6),
    strip.background = element_rect(fill = "white", color = "black"),
    plot.margin = margin(t = 0.1, r = 0.1, b = 0.1, l = 0.1, unit = "cm"),
    aspect.ratio = 1
  )
p1

ggsave(
  plot = p1,
  file = "../../figures/nll_landscape.svg",
  width = 1000, height = 1000, units = "px",
  scale = 3
)
