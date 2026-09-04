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
mod5 <- cmdstan_model('d5_rc_ddm_generative.stan')
mod6 <- cmdstan_model('d6_rc_ddm_distilled.stan')

res5 <- numeric(10)
res6 <- numeric(10)

for (i in 1:10) {
  cat(sprintf("\n--- Iteration %d ---\n", i))
  
  fit5 <- mod5$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd5 <- mean(fit5$draws(variables = "sum_log_lik"))
  res5[i] <- elpd5
  
  fit6 <- mod6$pathfinder(data = data_N15, num_paths = 4, single_path_draws = 1000, refresh=0)
  elpd6 <- mean(fit6$draws(variables = "sum_log_lik"))
  res6[i] <- elpd6
  
  cat(sprintf("D5 (Generative) ELPD: %.4f\n", elpd5))
  cat(sprintf("D6 (Distilled) ELPD:  %.4f\n", elpd6))
}

cat("\n=== SUMMARY ===\n")
for (i in 1:10) {
  cat(sprintf("Iter %02d | D5: %.4f | D6: %.4f\n", i, res5[i], res6[i]))
}
