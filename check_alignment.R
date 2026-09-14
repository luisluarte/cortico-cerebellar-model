d <- readRDS('data/dataset_A.rds')
targets <- readRDS('src/r/distillation_targets_v2.rds')

# targets drops the first trial of each block/subject.
# We want to get the exact X matrix from d that matches targets.
d$subj_num <- as.numeric(as.factor(d$participant_id))

# Create a unique trial ID in both
d$trial_id <- paste(d$subj_num, d$nt, sep="_")

# How can we map targets to d?
# targets doesn't have 'nt' (trial number). But it has lag_reward, lag_resp, etc.
# Actually, the simplest way is to realize that targets was created by just dropping trials where lag is 0?
# No, let's look at export_distillation_targets.R
# mask <- c(FALSE, rep(TRUE, length(idx)-1))
# So it literally just drops the FIRST row of each subject in stan_data!

cat("d rows:", nrow(d), "\n")
cat("targets rows:", length(targets$labels), "\n")
