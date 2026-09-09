
library(jsonlite)
stan_data <- read_json("../../data/stan_data.json", simplifyVector = TRUE)
rnn_phase1_subjs <- readRDS("rnn_phase1_subjects.rds")

all_subjs <- unique(stan_data$subj)
cat("Total subjs: ", length(all_subjs), "\n")
cat("Phase 1 subjs: ", length(rnn_phase1_subjs), "\n")

oob_subjs <- setdiff(all_subjs, rnn_phase1_subjs)
cat("OOB subjs: ", length(oob_subjs), "\n")

saveRDS(oob_subjs, "rnn_oob_subjects.rds")
