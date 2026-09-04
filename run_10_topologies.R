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
  soft_p = data$soft_p[idx],
  min_RT = min(data$RT[idx])
)

cat("Compiling Models...\n")
mod8 <- cmdstan_model('d8_universal_generative.stan')
mod9 <- cmdstan_model('d9_universal_distilled.stan')

res8 <- numeric(10)
res9 <- numeric(10)

cat("Starting 10-Topology Exploration...\n")

for (topo in 1:10) {
  cat(sprintf("\n--- Evaluating Topology %d ---\n", topo))
  data_N15$topo_id <- topo
  
  fit8 <- mod8$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd8 <- mean(fit8$draws(variables = "sum_log_lik"))
  res8[topo] <- elpd8
  
  fit9 <- mod9$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd9 <- mean(fit9$draws(variables = "sum_log_lik"))
  res9[topo] <- elpd9
  
  cat(sprintf("Topo %d - Generative ELPD: %.4f\n", topo, elpd8))
  cat(sprintf("Topo %d - Distilled ELPD:  %.4f\n", topo, elpd9))
}

cat("\n=== 10-TOPOLOGY SUMMARY (N=15) ===\n")
cat("ID | Description                | Generative | Distilled (Hybrid)\n")
cat("-----------------------------------------------------------------\n")
desc <- c(
  "Boundary Mismatch (a_dyn)  ",
  "Drift Mismatch (v_t)       ",
  "NDT Mismatch (ndt_dyn)     ",
  "Dual Mismatch (a_dyn + v_t)",
  "Value Bonus (Q + Cb)       ",
  "No Fading Memory           ",
  "No Golgi Gating            ",
  "Pure Purkinje Drift        ",
  "Exponential Boundary       ",
  "Inverse Caution Boundary   "
)
for (i in 1:10) {
  cat(sprintf("%02d | %s | %10.2f | %10.2f\n", i, desc[i], res8[i], res9[i]))
}
