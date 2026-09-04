library(cmdstanr)
library(jsonlite)

data <- read_json('data/stan_data_N30_distilled.json', simplifyVector = TRUE)
data$min_RT <- min(data$RT)

cat("Compiling D7 Dynamic Distilled RC-DDM...\n")
mod7 <- cmdstan_model('d7_rc_ddm_dynamic_distilled.stan')
cat("Running D7 Dynamic Distilled RC-DDM Pathfinder...\n")
fit7 <- mod7$pathfinder(data = data, num_paths = 4, single_path_draws = 1000)
elpd_7 <- mean(fit7$draws(variables = "sum_log_lik"))

cat(sprintf("\n=== Cerebellar RC-DDM Topology (N=30) ===\n"))
cat(sprintf("D7 Dynamic Distilled ELPD: %.2f\n", elpd_7))
cat(sprintf("=========================================\n"))
