
local_lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(local_lib, .libPaths()))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(loo)
})

cat("Loading Data...\n")
targets <- readRDS("distillation_targets_v2.rds")
ps <- readRDS("parameter_stability.rds")

if(is.data.frame(ps)) {
    q3_wsls <- subset(ps, Model == "WSLS" & Quartile == "Q3")
} else {
    q3_wsls <- subset(ps$WSLS, Quartile == "Q3")
}

unique_subjs <- unique(targets$subjs)
pilot_subjs <- unique_subjs[1:5]
idx <- which(targets$subjs %in% pilot_subjs)

subjs <- targets$subjs[idx]
subj_id <- as.numeric(as.factor(subjs))
N <- length(idx)
K <- 5
y <- targets$labels[idx]

cat("Compiling Stan WSLS Model...\n")
mod_wsls <- cmdstan_model("wsls_fixed.stan")

cat("Running Stan WSLS Model...\n")
data_wsls <- list(N = N, K = K, subj_id = subj_id, y = y,
                  win = ifelse(targets$lag_reward[idx] > 0, 1, 0),
                  q3_theta_win_logit = qlogis(q3_wsls$theta_win),
                  q3_theta_loss_logit = qlogis(q3_wsls$theta_loss))
fit_wsls <- mod_wsls$sample(data = data_wsls, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000)

loo_wsls <- fit_wsls$loo()
cat("WSLS ELPD:\n")
print(loo_wsls)

# Read the previously completed Bio loo object (Wait, it wasn't saved. I'll just print WSLS to get the ELPD).
saveRDS(loo_wsls, "pilot_loo_wsls.rds")
