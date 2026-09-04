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

cat("Compiling Models...\n")
mod10 <- cmdstan_model('d10_ndt_exploration_generative.stan')
mod11 <- cmdstan_model('d11_ndt_exploration_distilled.stan')

res10 <- numeric(20)
res11 <- numeric(20)

cat("Starting 20-Topology NDT Exploration...\n")

for (topo in 1:20) {
  cat(sprintf("\n--- Evaluating NDT Topology %d ---\n", topo))
  data_N15$topo_id <- topo
  
  fit10 <- mod10$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd10 <- mean(fit10$draws(variables = "sum_log_lik"))
  res10[topo] <- elpd10
  
  fit11 <- mod11$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd11 <- mean(fit11$draws(variables = "sum_log_lik"))
  res11[topo] <- elpd11
  
  cat(sprintf("Topo %d - Generative ELPD: %.4f\n", topo, elpd10))
  cat(sprintf("Topo %d - Distilled ELPD:  %.4f\n", topo, elpd11))
}

cat("\n=== 20-TOPOLOGY NDT SUMMARY (N=15) ===\n")
cat("ID | Description                      | Generative | Distilled (Hybrid)\n")
cat("-----------------------------------------------------------------------\n")
desc <- c(
  "Base Linear Sqrt Mismatch      ",
  "Absolute Mismatch              ",
  "Squared Mismatch               ",
  "Tanh Saturated Mismatch        ",
  "Exponential NDT Mapping        ",
  "Logarithmic NDT Mapping        ",
  "Independent Bandit Mismatch    ",
  "Pure Cerebellar Magnitude      ",
  "Positive Cortical Surprise     ",
  "Negative Cortical Surprise     ",
  "Softmax Granule Input          ",
  "Global Golgi Gating            ",
  "Leaky Purkinje (Forgetting)    ",
  "Non-linear Purkinje Readout    ",
  "Cerebellar-Gated Q-Learning    ",
  "Cerebellar-Boosted Q-Learning  ",
  "RPE-Gated Purkinje Plasticity  ",
  "Dual Modulator (NDT & Bound)   ",
  "Reward-Only Purkinje Learning  ",
  "Anti-Hebbian Purkinje Learning "
)
for (i in 1:20) {
  cat(sprintf("%02d | %s | %10.2f | %10.2f\n", i, desc[i], res10[i], res11[i]))
}
