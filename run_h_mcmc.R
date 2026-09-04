library(cmdstanr)
library(jsonlite)

data <- read_json('data/stan_data_N30_distilled.json', simplifyVector = TRUE)

n_subj_target <- 30
valid_subjs <- unique(data$subj)[1:n_subj_target]
idx <- data$subj %in% valid_subjs

data_N30 <- list(
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

cat("Compiling Hierarchical MCMC Models...\\n")
mod_gen <- cmdstan_model('h_topo6_generative.stan')
mod_dist <- cmdstan_model('h_topo6_distilled.stan')

cat("Starting Generative MCMC...\\n")
fit_gen <- mod_gen$sample(
  data = data_N30,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  refresh = 10
)

cat("Starting Distilled MCMC...\\n")
fit_dist <- mod_dist$sample(
  data = data_N30,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  refresh = 10
)

gen_elpd <- mean(fit_gen$draws(variables = "sum_log_lik"))
dist_elpd <- mean(fit_dist$draws(variables = "sum_log_lik"))

cat(sprintf("\\n\\n=== FINAL HIERARCHICAL MCMC ELPD (N=30) ===\\n"))
cat(sprintf("Generative (Empirical): %.4f\\n", gen_elpd))
cat(sprintf("Distilled (Hybrid):     %.4f\\n", dist_elpd))

fit_gen$save_object("fit_gen_h_topo6.RDS")
fit_dist$save_object("fit_dist_h_topo6.RDS")
