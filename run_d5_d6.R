library(cmdstanr)
library(jsonlite)

data <- read_json('data/stan_data_N30_distilled.json', simplifyVector = TRUE)
data$min_RT <- min(data$RT)

cat("Compiling D5 Generative RC-DDM...\n")
mod5 <- cmdstan_model('d5_rc_ddm_generative.stan')
cat("Running D5 Generative RC-DDM Pathfinder...\n")
fit5 <- mod5$pathfinder(data = data, num_paths = 4, single_path_draws = 1000)
elpd_5 <- mean(fit5$draws(variables = "sum_log_lik"))

cat("Compiling D6 Distilled RC-DDM...\n")
mod6 <- cmdstan_model('d6_rc_ddm_distilled.stan')
cat("Running D6 Distilled RC-DDM Pathfinder...\n")
fit6 <- mod6$pathfinder(data = data, num_paths = 4, single_path_draws = 1000)
elpd_6 <- mean(fit6$draws(variables = "sum_log_lik"))

cat(sprintf("\n=== Cerebellar RC-DDM Topology (N=30) ===\n"))
cat(sprintf("Generative (Unconstrained) ELPD: %.2f\n", elpd_5))
cat(sprintf("Hybrid Distilled ELPD: %.2f\n", elpd_6))
cat(sprintf("=========================================\n"))
