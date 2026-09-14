targets <- readRDS("src/r/distillation_targets_v2.rds")
post_df <- readRDS("results/factorial_fixed_10k_A.rds")
cat("Targets subjs:", length(unique(targets$subjs)), "\n")
cat("Post traces:", length(post_df[["bio_dist"]][["traces"]]), "\n")
