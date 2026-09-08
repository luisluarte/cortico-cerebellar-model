
t50 <- readRDS('distillation_targets_v2.rds')
t100 <- readRDS('distillation_targets_N100.rds')
cat("t50 subjects:", length(unique(t50$subjs)), "\n")
cat("t100 subjects:", length(unique(t100$subjs)), "\n")

s_shared <- unique(t50$subjs)[1]
idx50 <- which(t50$subjs == s_shared)
idx100 <- which(t100$subjs == s_shared)
cat("Subject", s_shared, "trials in t50:", length(idx50), "\n")
cat("Subject", s_shared, "trials in t100:", length(idx100), "\n")

cat("t50 X sum:", sum(t50$X[idx50,]), "\n")
cat("t100 X sum:", sum(t100$X[idx100,]), "\n")
cat("t50 lag_reward sum:", sum(t50$lag_reward[idx50]), "\n")
cat("t100 lag_reward sum:", sum(t100$lag_reward[idx100]), "\n")
