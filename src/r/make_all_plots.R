# setup ----

# set local lib path
local_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# set download preferences
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://cloud.r-project.org"
  options(repos = r)
})

# install package manager
cat("\n############ installing pacman #############\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman", lib = local_lib)
}

cat("\n###### loading libs #########\n")
pacman::p_load(
  tidyverse,
  torch,
  jsonlite,
  pROC,
  cli,
  httpgd,
  ggpubr
)

if (interactive() && requireNamespace("httpgd", quietly = TRUE)) {
  try(httpgd::hgd(port = 8288, host = "127.0.0.1", token = FALSE), silent = TRUE)
  cli::cli_alert_info("httpgd viewer ready at: http://127.0.0.1:8288/live")
}

cli::cli_h2("setting workspace dir")
if (dir.exists("src/r")) {
  setwd("src/r")
}
cli::cli_text(getwd())

# plots ----

# super duper theme
theme_arxiv_physics <- function(base_size = 12, base_family = "") {
  ggpubr::theme_pubr(base_size = base_size, base_family = base_family, border = TRUE) %+replace%
    theme(
      # Full enclosed box frame (standard in APS/PRL)
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
      axis.line = element_blank(), # Handled cleanly by panel.border

      # Pure white background, strip out internal gridlines
      panel.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),

      # Crisp scientific ticks
      axis.ticks = element_line(color = "black", linewidth = 0.6),
      axis.ticks.length = unit(0.18, "cm"),

      # Typography
      axis.title = element_text(face = "bold", size = rel(1.05)),
      axis.text = element_text(color = "black", size = rel(0.95)),

      # Legend formatting
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.key = element_blank(),
      legend.title = element_text(face = "bold", size = rel(0.9)),

      # Subpanel facet strips (e.g., (a), (b))
      strip.background = element_rect(fill = "grey92", color = "black", linewidth = 0.8),
      strip.text = element_text(face = "bold", size = rel(1.0), margin = margin(4, 4, 4, 4))
    )
}


# for font size
scale_factor <- 3


p1_data <- read_rds("../../results/rnn_results.rds")

p1 <- p1_data %>%
  pivot_longer(
    cols = everything(),
    names_to = "metrics", values_to = "values"
  ) %>%
  ggplot(aes(
    metrics, values
  )) +
  geom_boxplot(outlier.shape = NA) +
  geom_point(shape = 21, size = 5) +
  theme_arxiv_physics() +
  ylab("") +
  xlab("")
p1

ggsave(
  plot = p1,
  "../../figures/rnn_performance_boxplot.pdf",
  width = 20,
  height = 20,
  dpi = 600
)
