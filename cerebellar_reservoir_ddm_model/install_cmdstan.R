install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", "https://cloud.r-project.org"))
library(cmdstanr)
check_cmdstan_toolchain(fix = TRUE)
install_cmdstan(cores = 32)
