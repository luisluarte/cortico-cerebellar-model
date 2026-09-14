targets <- readRDS("src/r/distillation_targets_v2.rds")
cat("Summary of labels:\n")
print(table(targets$labels))
cat("\nAre labels exactly 0 and 1?\n")
print(unique(targets$labels))

# Let's see if labels correlate with empirical switching
d_emp <- read.csv("data/behavioral_compilate.csv", stringsAsFactors = FALSE)
cat("\nEmpirical Resp column table:\n")
print(table(d_emp$Resp))
