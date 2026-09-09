
library(jsonlite)
targets <- readRDS("distillation_targets_v2.rds")
stan_data <- read_json("../../data/stan_data.json", simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

all_true_rt <- numeric()
for (s in rnn_phase1_subjs) {
    idx <- which(stan_data$subj == s)
    mask <- c(FALSE, rep(TRUE, length(idx)-1))
    all_true_rt <- c(all_true_rt, stan_data$RT[idx][mask])
}

cat("Length of targets$labels: ", length(targets$labels), "\n")
cat("Length of all_true_rt: ", length(all_true_rt), "\n")

targets$true_rt <- all_true_rt
saveRDS(targets, "distillation_targets_v2_empirical.rds")
cat("Saved empirical targets with true RT to distillation_targets_v2_empirical.rds\n")
