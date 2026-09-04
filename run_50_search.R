library(cmdstanr)
library(jsonlite)

data <- read_json('data/stan_data_N30_distilled.json', simplifyVector = TRUE)

n_subj_target <- 15
valid_subjs <- unique(data$subj)[1:n_subj_target]
idx <- data$subj %in% valid_subjs

data_N15 <- list(
  N = sum(idx),
  N_subj = n_subj_target,
  subj = as.numeric(factor(data$subj[idx])),
  Bd1 = data$Bd1[idx],
  Bd2 = data$Bd2[idx],
  Resp = data$Resp[idx],
  Reward = data$Reward[idx],
  RT = data$RT[idx],
  soft_p = data$soft_p[idx]
)

cat("Compiling Combinatorial Models...\n")
mod12 <- cmdstan_model('d12_combinatorial_generative.stan')
mod13 <- cmdstan_model('d13_combinatorial_distilled.stan')

# Generate 50 random topologies
set.seed(42)
topologies <- data.frame(
  id = 1:50,
  m_type = sample(c(1, 3, 4, 5), 50, replace=TRUE), # Skipping 2 (abs) to avoid cusps
  target_type = sample(1:4, 50, replace=TRUE),
  memory_type = sample(1:2, 50, replace=TRUE),
  golgi_type = sample(1:2, 50, replace=TRUE),
  lr_type = sample(1:3, 50, replace=TRUE),
  gen_elpd = NA_real_,
  dist_elpd = NA_real_
)

cat("Starting 50-Topology Search...\n")

for (i in 1:50) {
  cat(sprintf("\n--- Evaluating Combinatorial Topology %d ---\n", i))
  
  data_N15$m_type <- topologies$m_type[i]
  data_N15$target_type <- topologies$target_type[i]
  data_N15$memory_type <- topologies$memory_type[i]
  data_N15$golgi_type <- topologies$golgi_type[i]
  data_N15$lr_type <- topologies$lr_type[i]
  
  tryCatch({
    fit12 <- mod12$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
    topologies$gen_elpd[i] <- mean(fit12$draws(variables = "sum_log_lik"))
  }, error = function(e) { cat("Topo", i, "Generative Failed!\n") })
  
  tryCatch({
    fit13 <- mod13$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
    topologies$dist_elpd[i] <- mean(fit13$draws(variables = "sum_log_lik"))
  }, error = function(e) { cat("Topo", i, "Distilled Failed!\n") })
  
  cat(sprintf("Topo %d - Generative ELPD: %.4f\n", i, topologies$gen_elpd[i]))
  cat(sprintf("Topo %d - Distilled ELPD:  %.4f\n", i, topologies$dist_elpd[i]))
  
  # Save continuously
  write.csv(topologies, "combinatorial_50_results.csv", row.names=FALSE)
}
cat("\nSearch Complete. Results saved to combinatorial_50_results.csv\n")
